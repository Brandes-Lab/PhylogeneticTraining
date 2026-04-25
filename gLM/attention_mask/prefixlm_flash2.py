"""
PrefixLM with Flash Attention 2 for ModernBERT.

Instead of a dense 4D mask (O(T²) memory), this uses two flash_attn_varlen_func
calls per layer:
    Call 1 — Prefix (bidirectional): Q=prefix, K=prefix, V=prefix, causal=False
    Call 2 — Suffix (causal + cross): Q=suffix, K=full, V=full, causal=True
        Because seqlen_q < seqlen_k and causal=True, flash attention uses
        bottom-right alignment: suffix query i attends to keys [0 .. P+i].

Global layers  → full attention window (no window_size constraint)
Local layers   → windowed attention (window_size from config.local_attention)

Optimisations vs naive implementation:
  - cu_seqlens built once, reused every layer
  - Pack indices (prefix/suffix/full) built once as flat CUDA index tensors;
    packing is a single tensor index op per region — no Python loop per layer
  - Both RoPE (global + local theta) built in one fused call before the loop
  - layer_is_global is a plain bool list; per-layer cost is one list lookup
"""

import torch
from torch.nn import CrossEntropyLoss

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb


# =============================================================================
# RoPE — fused build for both global and local theta
# =============================================================================

def build_rope_cos_sin_both(seq_len, head_dim, theta_global, theta_local, device, dtype):
    """
    Build RoPE cos/sin tensors for both global and local theta in one pass.
    Avoids duplicating the arange/outer/cat work.

    Returns:
        (cos_global, sin_global), (cos_local, sin_local)
        each tensor shaped (1, seq_len, head_dim).
    """
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)  # (T,)
    half_dim  = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)  # (D/2,)

    def _make(theta):
        inv_freq = 1.0 / (theta ** (half_dim / head_dim))          # (D/2,)
        freqs    = torch.outer(positions, inv_freq)                  # (T, D/2)
        c = torch.cat([freqs.cos(), freqs.cos()], dim=-1)           # (T, D)
        s = torch.cat([freqs.sin(), freqs.sin()], dim=-1)           # (T, D)
        return c.to(dtype).unsqueeze(0), s.to(dtype).unsqueeze(0)   # (1, T, D)

    return _make(theta_global), _make(theta_local)


# =============================================================================
# Layer-type list — read from config once before the layer loop
# =============================================================================

def build_layer_type_list(config):
    """
    Return a plain bool list [is_global, ...] of length num_hidden_layers.

    Priority:
      1. config.layer_types  — list ModernBERT builds from
                               global_attn_every_n_layers at config init.
                               Handles any regular or irregular pattern.
      2. config.global_attn_every_n_layers  — fallback recomputation.
      3. All-global (safe fallback; user should always set one of the above).
    """
    if hasattr(config, "layer_types") and config.layer_types is not None:
        return [lt == "global" for lt in config.layer_types]

    if hasattr(config, "global_attn_every_n_layers"):
        n = config.global_attn_every_n_layers
        return [((i + 1) % n == 0) for i in range(config.num_hidden_layers)]

    return [True] * config.num_hidden_layers


# =============================================================================
# Pack indices — built ONCE, reused every layer
# =============================================================================

def build_pack_indices(B, T, seq_lens_list, prefix_lens_list, suffix_lens_list, device):
    """
    Precompute flat CUDA index tensors for prefix, suffix, and full regions.

    Each index tensor contains positions into the flattened (B*T) token
    dimension. Inside the layer loop, packing becomes a single tensor
    index op — no Python for-loop overhead per layer.

    Returns:
        prefix_idx : (total_prefix_tokens,)
        suffix_idx : (total_suffix_tokens,)
        full_idx   : (total_tokens,)
    """
    prefix_parts = []
    suffix_parts = []
    full_parts   = []

    for i in range(B):
        base  = i * T
        plen  = prefix_lens_list[i]
        total = seq_lens_list[i]

        prefix_parts.append(torch.arange(base,        base + plen,  device=device))
        suffix_parts.append(torch.arange(base + plen, base + total, device=device))
        full_parts.append(  torch.arange(base,        base + total, device=device))

    return (
        torch.cat(prefix_parts),  # (total_prefix_tokens,)
        torch.cat(suffix_parts),  # (total_suffix_tokens,)
        torch.cat(full_parts),    # (total_full_tokens,)
    )


# =============================================================================
# Unpack closure — reassemble (B, T, H, D) from flat prefix + suffix tensors
# =============================================================================

def build_unpack_fn(B, T, num_heads, head_dim,
                    prefix_lens_list, suffix_lens_list, device):
    """
    Returns a closure that unpacks flat prefix/suffix tensors back to (B, T, H, D).
    Prefix and suffix cumulative offsets are captured once in the closure so the
    per-call cost is just the slice assignments — no offset recomputation.
    """
    p_offsets = [0] * B
    s_offsets = [0] * B
    acc_p = acc_s = 0
    for i in range(B):
        p_offsets[i] = acc_p
        s_offsets[i] = acc_s
        acc_p += prefix_lens_list[i]
        acc_s += suffix_lens_list[i]

    def unpack(prefix_flat, suffix_flat):
        out = torch.zeros(B, T, num_heads, head_dim,
                          dtype=prefix_flat.dtype, device=device)
        for i in range(B):
            plen = prefix_lens_list[i]
            slen = suffix_lens_list[i]
            p0   = p_offsets[i]
            s0   = s_offsets[i]
            out[i, :plen]                = prefix_flat[p0 : p0 + plen]
            if slen > 0:
                out[i, plen : plen + slen] = suffix_flat[s0 : s0 + slen]
        return out

    return unpack


# =============================================================================
# Flash attention calls — global and local
# =============================================================================

def _global_two_call(
    q_pre, k_pre, v_pre,
    q_suf, k_ful, v_ful,
    cu_prefix, cu_full, cu_suffix_q,
    max_prefix, max_full, max_suffix,
    dropout_p,
):
    """Full (unbounded) attention for global layers."""
    out_prefix = flash_attn_varlen_func(
        q_pre, k_pre, v_pre,
        cu_seqlens_q=cu_prefix,
        cu_seqlens_k=cu_prefix,
        max_seqlen_q=max_prefix,
        max_seqlen_k=max_prefix,
        dropout_p=dropout_p,
        causal=False,
    )
    out_suffix = flash_attn_varlen_func(
        q_suf, k_ful, v_ful,
        cu_seqlens_q=cu_suffix_q,
        cu_seqlens_k=cu_full,
        max_seqlen_q=max_suffix,
        max_seqlen_k=max_full,
        dropout_p=dropout_p,
        causal=True,
    )
    return out_prefix, out_suffix


def _local_two_call(
    q_pre, k_pre, v_pre,
    q_suf, k_ful, v_ful,
    cu_prefix, cu_full, cu_suffix_q,
    max_prefix, max_full, max_suffix,
    local_window,
    dropout_p,
):
    """
    Windowed attention for local layers.

    window_size=(left, right): number of tokens to attend to on each side.
    Prefix: symmetric window, non-causal.
    Suffix: left-only window, causal — each suffix query looks back at most
            local_window tokens into the full prefix+suffix sequence.
    """
    half = local_window // 2

    out_prefix = flash_attn_varlen_func(
        q_pre, k_pre, v_pre,
        cu_seqlens_q=cu_prefix,
        cu_seqlens_k=cu_prefix,
        max_seqlen_q=max_prefix,
        max_seqlen_k=max_prefix,
        dropout_p=dropout_p,
        causal=False,
        window_size=(half, half),
    )
    out_suffix = flash_attn_varlen_func(
        q_suf, k_ful, v_ful,
        cu_seqlens_q=cu_suffix_q,
        cu_seqlens_k=cu_full,
        max_seqlen_q=max_suffix,
        max_seqlen_k=max_full,
        dropout_p=dropout_p,
        causal=True,
        window_size=(local_window, 0),
    )
    return out_prefix, out_suffix


# =============================================================================
# Encoder forward
# =============================================================================

def run_encoder_flash(model, input_ids, prefix_lengths, device):
    """
    Full encoder forward using Flash Attention 2, respecting
    ModernBERT's local/global layer pattern from the model config.
    """
    base_model = model.module if hasattr(model, "module") else model
    encoder    = base_model.model
    config     = encoder.config

    B, T      = input_ids.shape
    num_heads = config.num_attention_heads
    head_dim  = config.hidden_size // num_heads

    # ── Sequence lengths ─────────────────────────────────────────────
    padding_mask = (input_ids != config.pad_token_id)
    seq_lens     = padding_mask.sum(dim=1).int()
    prefix_lens  = prefix_lengths.int()
    suffix_lens  = seq_lens - prefix_lens

    # ── cu_seqlens — built ONCE, reused every layer ──────────────────
    cu_full = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_full[1:]     = seq_lens.cumsum(0)

    cu_prefix = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_prefix[1:]   = prefix_lens.cumsum(0)

    cu_suffix_q = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_suffix_q[1:] = suffix_lens.cumsum(0)

    max_full   = seq_lens.max().item()
    max_prefix = prefix_lens.max().item()
    max_suffix = suffix_lens.max().item()

    # Python lists for slice arithmetic (avoids .item() in the loop)
    seq_lens_list    = seq_lens.tolist()
    prefix_lens_list = prefix_lens.tolist()
    suffix_lens_list = suffix_lens.tolist()

    # ── Pack indices — built ONCE as flat CUDA tensors ────────────────
    # Per-layer packing = 3 index ops (prefix/suffix/full), no Python loop
    prefix_idx, suffix_idx, full_idx = build_pack_indices(
        B, T, seq_lens_list, prefix_lens_list, suffix_lens_list, device
    )

    # ── Unpack closure — cumulative offsets computed once ─────────────
    unpack = build_unpack_fn(
        B, T, num_heads, head_dim,
        prefix_lens_list, suffix_lens_list, device
    )

    # ── Layer type list — one list-index lookup per layer ─────────────
    layer_is_global = build_layer_type_list(config)
    local_window    = getattr(config, "local_attention", 512)

    # ── Embeddings ────────────────────────────────────────────────────
    hidden_states = encoder.embeddings(input_ids=input_ids)

    # ── RoPE — both thetas built in ONE fused call ────────────────────
    theta_global = config.global_rope_theta
    theta_local  = getattr(config, "local_rope_theta", 10000.0)

    (cos_global, sin_global), (cos_local, sin_local) = build_rope_cos_sin_both(
        T, head_dim, theta_global, theta_local, device, hidden_states.dtype
    )

    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    dropout_p    = config.attention_dropout if model.training else 0.0

    # ── Layer loop ────────────────────────────────────────────────────
    for layer_idx, layer in enumerate(encoder.layers):

        # One list-index lookup — no arithmetic in the hot path
        is_global = layer_is_global[layer_idx]

        attn   = layer.attn
        normed = layer.attn_norm(hidden_states)

        # QKV projection
        qkv = attn.Wqkv(normed)
        qkv = qkv.view(B, T, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)

        # RoPE with correct theta for this layer type
        cos = cos_global if is_global else cos_local
        sin = sin_global if is_global else sin_local

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        q_t, k_t = apply_rotary_pos_emb(
            q_t, k_t, cos, sin,
            position_ids=position_ids,
            unsqueeze_dim=1,
        )
        q = q_t.transpose(1, 2).contiguous().to(torch.bfloat16)
        k = k_t.transpose(1, 2).contiguous().to(torch.bfloat16)
        v = v.contiguous().to(torch.bfloat16)

        # Pack — single CUDA index op per region, zero Python loop overhead
        q_flat = q.reshape(B * T, num_heads, head_dim)
        k_flat = k.reshape(B * T, num_heads, head_dim)
        v_flat = v.reshape(B * T, num_heads, head_dim)

        q_pre = q_flat[prefix_idx];  k_pre = k_flat[prefix_idx];  v_pre = v_flat[prefix_idx]
        q_suf = q_flat[suffix_idx];  k_ful = k_flat[full_idx];    v_ful = v_flat[full_idx]

        # Two-call flash attention — global or local
        if is_global:
            out_prefix, out_suffix = _global_two_call(
                q_pre, k_pre, v_pre,
                q_suf, k_ful, v_ful,
                cu_prefix, cu_full, cu_suffix_q,
                max_prefix, max_full, max_suffix,
                dropout_p,
            )
        else:
            out_prefix, out_suffix = _local_two_call(
                q_pre, k_pre, v_pre,
                q_suf, k_ful, v_ful,
                cu_prefix, cu_full, cu_suffix_q,
                max_prefix, max_full, max_suffix,
                local_window,
                dropout_p,
            )

        # Unpack → project → residual
        attn_out = unpack(out_prefix, out_suffix)            # (B, T, H, D)
        attn_out = attn_out.reshape(B, T, -1).to(hidden_states.dtype)
        attn_out = attn.out_drop(attn.Wo(attn_out))
        hidden_states = hidden_states + attn_out

        # MLP block — unchanged from standard ModernBERT
        hidden_states = hidden_states + layer.mlp(layer.mlp_norm(hidden_states))

    hidden_states = encoder.final_norm(hidden_states)
    return hidden_states


# =============================================================================
# Full forward pass
# =============================================================================

def prefixlm_forward_flash(model, batch, device):
    """
    Full forward pass using Flash Attention 2 with local/global layer routing.
    Drop-in replacement for prefixlm_forward.
    """
    base_model = model.module if hasattr(model, "module") else model

    input_ids      = batch["input_ids"].to(device)
    labels         = batch["labels"].to(device)
    prefix_lengths = batch["prefix_lengths"].to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden_states = run_encoder_flash(model, input_ids, prefix_lengths, device)
        logits        = base_model.decoder(base_model.head(hidden_states))

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = CrossEntropyLoss(ignore_index=-100)(
            shift_logits.view(-1, base_model.config.vocab_size),
            shift_labels.view(-1),
        )

    return loss, logits