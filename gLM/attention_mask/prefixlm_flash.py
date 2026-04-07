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

def prefixlm_flash_attention(q, k, v, prefix_lengths, seq_lens, dropout_p=0.0):
    """
    PrefixLM attention using two flash_attn_varlen_func calls.

    Args:
        q, k, v:          (B, T, num_heads, head_dim) — with RoPE already applied
        prefix_lengths:   (B,) int32 — length of [CLS] + seq2 + [SEP]
        seq_lens:         (B,) int32 — actual sequence lengths (excluding padding)
        dropout_p:        float — attention dropout (0 during eval)

    Returns:
        (B, T, num_heads, head_dim) — attention output, zeros at padding positions
    """
    B, T, H, D = q.shape
    device = q.device
    suffix_lens = seq_lens - prefix_lengths

    # --- Call 1: Prefix (bidirectional self-attention) ---
    prefix_q = [q[i, :prefix_lengths[i]] for i in range(B)]
    prefix_k = [k[i, :prefix_lengths[i]] for i in range(B)]
    prefix_v = [v[i, :prefix_lengths[i]] for i in range(B)]

    prefix_q_flat = torch.cat(prefix_q, dim=0)  # (total_prefix_tokens, H, D)
    prefix_k_flat = torch.cat(prefix_k, dim=0)
    prefix_v_flat = torch.cat(prefix_v, dim=0)

    cu_prefix = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_prefix[1:] = torch.cumsum(prefix_lengths, dim=0)
    max_prefix = prefix_lengths.max().item()

    out_prefix_flat = flash_attn_varlen_func(
        prefix_q_flat, prefix_k_flat, prefix_v_flat,
        cu_seqlens_q=cu_prefix,
        cu_seqlens_k=cu_prefix,
        max_seqlen_q=max_prefix,
        max_seqlen_k=max_prefix,
        dropout_p=dropout_p,
        causal=False,
    )

    # --- Call 2: Suffix (causal, attending to full sequence) ---
    # Q = suffix tokens only, K/V = full sequence (prefix + suffix)
    # Because seqlen_q (suffix) < seqlen_k (full) and causal=True,
    # flash attention aligns the causal mask to the bottom-right:
    #   suffix query i attends to key positions [0 .. prefix_len + i]
    out_suffix_flat = None
    if suffix_lens.max().item() > 0:
        suffix_q = [q[i, prefix_lengths[i]:seq_lens[i]] for i in range(B)]
        full_k = [k[i, :seq_lens[i]] for i in range(B)]
        full_v = [v[i, :seq_lens[i]] for i in range(B)]

        suffix_q_flat = torch.cat(suffix_q, dim=0)  # (total_suffix_tokens, H, D)
        full_k_flat = torch.cat(full_k, dim=0)       # (total_full_tokens, H, D)
        full_v_flat = torch.cat(full_v, dim=0)

        cu_suffix_q = torch.zeros(B + 1, dtype=torch.int32, device=device)
        cu_suffix_q[1:] = torch.cumsum(suffix_lens, dim=0)
        cu_full_k = torch.zeros(B + 1, dtype=torch.int32, device=device)
        cu_full_k[1:] = torch.cumsum(seq_lens, dim=0)
        max_suffix = suffix_lens.max().item()
        max_full = seq_lens.max().item()

        out_suffix_flat = flash_attn_varlen_func(
            suffix_q_flat, full_k_flat, full_v_flat,
            cu_seqlens_q=cu_suffix_q,
            cu_seqlens_k=cu_full_k,
            max_seqlen_q=max_suffix,
            max_seqlen_k=max_full,
            dropout_p=dropout_p,
            causal=True,  # bottom-right aligned
        )

    # --- Reassemble into (B, T, H, D) ---
    out = torch.zeros_like(q)
    p_offset = 0
    s_offset = 0
    for i in range(B):
        plen = prefix_lengths[i].item()
        slen = suffix_lens[i].item()
        out[i, :plen] = out_prefix_flat[p_offset : p_offset + plen]
        if slen > 0:
            out[i, plen : plen + slen] = out_suffix_flat[s_offset : s_offset + slen]
            s_offset += slen
        p_offset += plen

    return out


# =============================================================================
# Encoder forward with Flash Attention
# =============================================================================


def run_encoder_flash(model, input_ids, prefix_lengths, device):
    """
    Run the ModernBERT encoder with PrefixLM attention using Flash Attention 2.

    Bypasses both ModernBertModel.forward() AND ModernBertAttention.forward().
    Assumes all layers use the same attention type (full_attention) and the
    same RoPE theta — so cos/sin are computed once outside the layer loop.

    For each layer we manually:
        1. LayerNorm
        2. Compute Q, K, V via the layer's Wqkv weights
        3. Apply RoPE (precomputed cos/sin)
        4. Two-call flash attention (prefix bidirectional + suffix causal)
        5. Output projection via Wo
        6. Residual connection + MLP

    Args:
        model:          ModernBertForMaskedLM
        input_ids:      (B, T) on device
        prefix_lengths: (B,) on device
        device:         torch device

    Returns:
        hidden_states: (B, T, hidden_size) after final norm
    """
    if not FLASH_ATTN_AVAILABLE:
        raise ImportError(
            "flash_attn is required for run_encoder_flash. "
            "Install with: pip install flash-attn --no-build-isolation"
        )

    base_model = model.module if hasattr(model, "module") else model
    encoder = base_model.model
    config = encoder.config
    B, T = input_ids.shape
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    # Padding info
    padding_mask = (input_ids != config.pad_token_id)
    seq_lens = padding_mask.sum(dim=1).int()
    prefix_lengths_i32 = prefix_lengths.int()

    # Embeddings
    hidden_states = encoder.embeddings(input_ids=input_ids)

    theta = config.global_rope_theta
    cos, sin = build_rope_cos_sin(
        seq_len=T,
        head_dim=head_dim,
        theta=theta,
        device=device,
        dtype=hidden_states.dtype,
    )
    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

    dropout_p = config.attention_dropout if model.training else 0.0

    for layer in encoder.layers:
        attn = layer.attn

        # 1. Pre-attention LayerNorm
        normed = layer.attn_norm(hidden_states)

        # 2. Compute Q, K, V
        qkv = attn.Wqkv(normed)  # (B, T, 3 * num_heads * head_dim)
        qkv = qkv.view(B, T, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, num_heads, head_dim)

        # 3. Apply RoPE — the model's apply_rotary_pos_emb expects (B, H, T, D)
        # so we transpose in, apply, transpose back to (B, T, H, D) for FA2
        q_t = q.transpose(1, 2)  # (B, H, T, D)
        k_t = k.transpose(1, 2)
        q_t, k_t = apply_rotary_pos_emb(
            q_t, k_t, cos, sin, position_ids=position_ids, unsqueeze_dim=1
        )
        q = q_t.transpose(1, 2).contiguous()  # back to (B, T, H, D)
        k = k_t.transpose(1, 2).contiguous()

        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        # 4. Two-call flash attention
        attn_output = prefixlm_flash_attention(
            q, k, v,
            prefix_lengths=prefix_lengths_i32,
            seq_lens=seq_lens,
            dropout_p=dropout_p,
        )

        # 5. Output projection + residual
        attn_output = attn_output.reshape(B, T, -1).contiguous()
        attn_output = attn_output.to(hidden_states.dtype)
        attn_output = attn.out_drop(attn.Wo(attn_output))
        hidden_states = hidden_states + attn_output

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