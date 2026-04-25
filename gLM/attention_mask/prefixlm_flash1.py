"""
PrefixLM with Flash Attention 2 for ModernBERT.

Instead of a dense 4D mask (O(T²) memory), this uses two flash_attn_varlen_func
calls per layer:
    Call 1 — Prefix (bidirectional): Q=prefix, K=prefix, V=prefix, causal=False
    Call 2 — Suffix (causal + cross): Q=suffix, K=full, V=full, causal=True
        Because seqlen_q < seqlen_k and causal=True, flash attention uses
        bottom-right alignment: suffix query i attends to keys [0 .. P+i].

This bypasses ModernBertAttention.forward() entirely — we manually call Wqkv,
apply RoPE (using the model's own rotary embedding module), run flash attention,
and apply Wo for each layer.

Memory: O(T) instead of O(T²) for the attention mask.
Speed: Flash attention's IO-aware algorithm, no materialized attention matrix.
"""

import torch
from torch.nn import CrossEntropyLoss

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb


def build_rope_cos_sin(seq_len, head_dim, theta, device, dtype):
    """
    Build RoPE cos/sin tensors compatible with apply_rotary_pos_emb
    for q,k shaped (B, H, T, D).
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )  # (D/2,)

    positions = torch.arange(seq_len, device=device, dtype=torch.float32)  # (T,)
    freqs = torch.outer(positions, inv_freq)  # (T, D/2)

    cos = freqs.cos()
    sin = freqs.sin()

    # expand from (T, D/2) -> (1, T, D)
    cos = torch.cat([cos, cos], dim=-1).to(dtype).unsqueeze(0)
    sin = torch.cat([sin, sin], dim=-1).to(dtype).unsqueeze(0)

    return cos, sin


# =============================================================================
# Two-call flash attention for PrefixLM
# =============================================================================

def prefixlm_flash_attention(q_prefix, k_prefix, v_prefix,
                              q_suffix, k_full, v_full,
                              cu_prefix, cu_full, cu_suffix_q,
                              max_prefix, max_full, max_suffix,
                              dropout_p=0.0):
    """
    Two flash_attn calls. All cu_seqlens and packed tensors are
    precomputed outside — no allocation or Python loops here.
    """
    out_prefix = flash_attn_varlen_func(
        q_prefix, k_prefix, v_prefix,
        cu_seqlens_q=cu_prefix,
        cu_seqlens_k=cu_prefix,
        max_seqlen_q=max_prefix,
        max_seqlen_k=max_prefix,
        dropout_p=dropout_p,
        causal=False,
    )
    out_suffix = flash_attn_varlen_func(
        q_suffix, k_full, v_full,
        cu_seqlens_q=cu_suffix_q,
        cu_seqlens_k=cu_full,
        max_seqlen_q=max_suffix,
        max_seqlen_k=max_full,
        dropout_p=dropout_p,
        causal=True,
    )
    return out_prefix, out_suffix


# =============================================================================
# Encoder forward with Flash Attention
# =============================================================================

def run_encoder_flash(model, input_ids, prefix_lengths, device):
    base_model = model.module if hasattr(model, "module") else model
    encoder = base_model.model
    config = encoder.config
    B, T = input_ids.shape
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    padding_mask = (input_ids != config.pad_token_id)
    seq_lens = padding_mask.sum(dim=1).int()
    prefix_lens = prefix_lengths.int()
    suffix_lens = seq_lens - prefix_lens

    # ── Precompute once, outside layer loop ──────────────────────────
    cu_full = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_full[1:] = seq_lens.cumsum(0)

    cu_prefix = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_prefix[1:] = prefix_lens.cumsum(0)

    cu_suffix_q = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_suffix_q[1:] = suffix_lens.cumsum(0)

    max_full   = seq_lens.max().item()
    max_prefix = prefix_lens.max().item()
    max_suffix = suffix_lens.max().item()

    # CPU lists for indexing — avoids .item() inside layer loop
    seq_lens_list    = seq_lens.tolist()
    prefix_lens_list = prefix_lens.tolist()
    suffix_lens_list = suffix_lens.tolist()
    # ─────────────────────────────────────────────────────────────────

    hidden_states = encoder.embeddings(input_ids=input_ids)

    theta = config.global_rope_theta
    cos, sin = build_rope_cos_sin(T, head_dim, theta, device, hidden_states.dtype)
    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    dropout_p = config.attention_dropout if model.training else 0.0

    def pack_prefix(x):
        # x: (B, T, H, D) → (total_prefix, H, D)
        return torch.cat([x[i, :prefix_lens_list[i]] for i in range(B)], dim=0)

    def pack_suffix(x):
        # x: (B, T, H, D) → (total_suffix, H, D)
        return torch.cat([x[i, prefix_lens_list[i]:seq_lens_list[i]] for i in range(B)], dim=0)

    def pack_full(x):
        # x: (B, T, H, D) → (total_full, H, D)
        return torch.cat([x[i, :seq_lens_list[i]] for i in range(B)], dim=0)

    def unpack(prefix_flat, suffix_flat):
        # → (B, T, H, D)
        out = torch.zeros(B, T, num_heads, head_dim,
                          dtype=prefix_flat.dtype, device=device)
        p_off = s_off = 0
        for i in range(B):
            plen = prefix_lens_list[i]
            slen = suffix_lens_list[i]
            out[i, :plen] = prefix_flat[p_off:p_off + plen]
            if slen > 0:
                out[i, plen:plen + slen] = suffix_flat[s_off:s_off + slen]
            p_off += plen
            s_off += slen
        return out

    for layer in encoder.layers:
        attn = layer.attn
        normed = layer.attn_norm(hidden_states)

        qkv = attn.Wqkv(normed)
        qkv = qkv.view(B, T, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        q_t, k_t = apply_rotary_pos_emb(q_t, k_t, cos, sin,
                                         position_ids=position_ids,
                                         unsqueeze_dim=1)
        q = q_t.transpose(1, 2).contiguous().to(torch.bfloat16)
        k = k_t.transpose(1, 2).contiguous().to(torch.bfloat16)
        v = v.contiguous().to(torch.bfloat16)

        # Pack — three helpers defined once above, no cu_seqlens built here
        q_pre = pack_prefix(q);  k_pre = pack_prefix(k);  v_pre = pack_prefix(v)
        q_suf = pack_suffix(q);  k_ful = pack_full(k);    v_ful = pack_full(v)

        out_prefix, out_suffix = prefixlm_flash_attention(
            q_pre, k_pre, v_pre,
            q_suf, k_ful, v_ful,
            cu_prefix, cu_full, cu_suffix_q,
            max_prefix, max_full, max_suffix,
            dropout_p=dropout_p,
        )

        attn_out = unpack(out_prefix, out_suffix)           # (B, T, H, D)
        attn_out = attn_out.reshape(B, T, -1).to(hidden_states.dtype)
        attn_out = attn.out_drop(attn.Wo(attn_out))
        hidden_states = hidden_states + attn_out

        # MLP — runs exactly as ModernBERT normally would
        hidden_states = hidden_states + layer.mlp(layer.mlp_norm(hidden_states))

    hidden_states = encoder.final_norm(hidden_states)
    return hidden_states

def prefixlm_forward_flash(model, batch, device):
    """
    Full forward pass using Flash Attention 2.
    Drop-in replacement for prefixlm_forward.
    """
    base_model = model.module if hasattr(model, "module") else model

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    prefix_lengths = batch["prefix_lengths"].to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden_states = run_encoder_flash(model, input_ids, prefix_lengths, device)
        logits = base_model.decoder(base_model.head(hidden_states))

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = CrossEntropyLoss(ignore_index=-100)(
            shift_logits.view(-1, base_model.config.vocab_size),
            shift_labels.view(-1),
        )

    return loss, logits