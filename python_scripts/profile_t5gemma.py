"""
Profile T5Gemma encoder vs decoder during training (forward + backward).

Runs three separate benchmarks:
  1. Encoder-only forward+backward (isolates encoder cost)
  2. Decoder-only forward+backward (with frozen encoder outputs, isolates decoder cost)
  3. Full model forward+backward (end-to-end, the actual training cost)

Usage:
    python python_scripts/profile_t5gemma.py \
        --tokenizer_path ./phylo_char_tokenizer_updated \
        --batch_size 128 \
        --seq_len 256
"""

import argparse
import torch
from torch.profiler import profile, record_function, ProfilerActivity

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gLM.models import ProteinT5GemmaModel
from gLM.tokenizers import PhyloTokenizerLoader


def make_dummy_batch(batch_size: int, seq_len: int, vocab_size: int, pad_id: int, device: torch.device):
    """Create a synthetic training batch matching phylo_encoder_decoder collator output."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)

    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    pad_mask = torch.rand(batch_size, seq_len, device=device) < 0.1
    labels[pad_mask] = -100

    decoder_attention_mask = (labels != -100).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "decoder_attention_mask": decoder_attention_mask,
    }


def benchmark_cuda(fn, warmup=3, repeats=10, label=""):
    """Run fn multiple times, return average GPU time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    print(f"  {label:40s} {avg:8.2f} ms  (std {std:.2f})")
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="./phylo_char_tokenizer_updated")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    parser.add_argument("--trace_output", type=str, default="profile_trace.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer and build model
    tokenizer = PhyloTokenizerLoader(args.tokenizer_path)
    model = ProteinT5GemmaModel(
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer,
        attn_implementation=args.attn_implementation,
    ).build()
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model = model.to(device).train()

    if not args.disable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    print(f"Gradient checkpointing: {'DISABLED' if args.disable_gradient_checkpointing else 'ENABLED'}")

    num_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.model.decoder.parameters())
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    print(f"Model parameters:   {num_params:,}")
    print(f"  Encoder:          {enc_params:,}")
    print(f"  Decoder:          {dec_params:,}")
    print(f"  LM Head:          {lm_head_params:,}")
    print(f"LM head shape: {model.lm_head.out_proj.weight.shape}")
    print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = make_dummy_batch(args.batch_size, args.seq_len, tokenizer.vocab_size, tokenizer.pad_token_id, device)

    # =========================================================================
    # Benchmark 1: Full model forward+backward (actual training step)
    # =========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARK 1: Full model (forward + backward + optimizer)")
    print("=" * 80)

    def full_step():
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()

    full_time = benchmark_cuda(full_step, label="Full train step")

    # =========================================================================
    # Benchmark 2: Encoder forward+backward only
    # =========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARK 2: Encoder only (forward + backward)")
    print("=" * 80)

    def encoder_fwd_bwd():
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            enc_out = model.model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            # Need a scalar loss to backward through encoder
            loss = enc_out.last_hidden_state.sum()
        loss.backward()

    enc_time = benchmark_cuda(encoder_fwd_bwd, label="Encoder fwd+bwd")

    # =========================================================================
    # Benchmark 3: Decoder forward+backward only (with detached encoder outputs)
    # =========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARK 3: Decoder + LM head + loss (forward + backward)")
    print("=" * 80)

    # Pre-compute encoder outputs and detach so encoder gets no gradients
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        frozen_enc = model.model.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
    frozen_hidden = frozen_enc.last_hidden_state.detach()

    from transformers.modeling_outputs import BaseModelOutput

    def decoder_fwd_bwd():
        optimizer.zero_grad()
        enc_output = BaseModelOutput(last_hidden_state=frozen_hidden)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                encoder_outputs=enc_output,
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                decoder_attention_mask=batch["decoder_attention_mask"],
            )
        outputs.loss.backward()

    dec_time = benchmark_cuda(decoder_fwd_bwd, label="Decoder+LMhead+loss fwd+bwd")

    # =========================================================================
    # Benchmark 4: Isolated forward passes (no backward)
    # =========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARK 4: Forward-only passes (no backward)")
    print("=" * 80)

    def encoder_fwd_only():
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model.model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

    def decoder_fwd_only():
        enc_output = BaseModelOutput(last_hidden_state=frozen_hidden)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(
                encoder_outputs=enc_output,
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                decoder_attention_mask=batch["decoder_attention_mask"],
            )

    def full_fwd_only():
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(**batch)

    enc_fwd = benchmark_cuda(encoder_fwd_only, label="Encoder forward only")
    dec_fwd = benchmark_cuda(decoder_fwd_only, label="Decoder+LMhead+loss fwd only")
    full_fwd = benchmark_cuda(full_fwd_only, label="Full model forward only")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  {'Component':<35s} {'Fwd+Bwd (ms)':>14s} {'Fwd only (ms)':>14s} {'Bwd only (ms)':>14s}")
    print(f"  {'-'*35} {'-'*14} {'-'*14} {'-'*14}")
    print(f"  {'Encoder':<35s} {enc_time:>14.2f} {enc_fwd:>14.2f} {enc_time - enc_fwd:>14.2f}")
    print(f"  {'Decoder + LM Head + Loss':<35s} {dec_time:>14.2f} {dec_fwd:>14.2f} {dec_time - dec_fwd:>14.2f}")
    print(f"  {'Full model':<35s} {full_time:>14.2f} {full_fwd:>14.2f} {full_time - full_fwd:>14.2f}")
    print()
    print(f"  Encoder share of fwd+bwd:   {enc_time / (enc_time + dec_time) * 100:.1f}%")
    print(f"  Decoder share of fwd+bwd:   {dec_time / (enc_time + dec_time) * 100:.1f}%")
    print()
    print(f"  Throughput (full step):     {args.batch_size / (full_time / 1000):.1f} samples/sec")
    print(f"  Throughput (encoder-only):  {args.batch_size / (enc_time / 1000):.1f} samples/sec (if you dropped the decoder)")


if __name__ == "__main__":
    main()
