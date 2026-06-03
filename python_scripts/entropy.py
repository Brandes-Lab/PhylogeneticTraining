"""
Score ClinVar variants across a RANGE of checkpoints and save:
- mutation log-probs
- mutation entropy
- OPTIONAL sequence-level log-prob + entropy

Usage:
python score_range_entropy.py \
    --checkpoint_dir /path/to/checkpoints \
    --tokenizer_path /path/to/tokenizer \
    --vep_csv input.csv \
    --output_csv output.csv \
    --start_step 1000 \
    --end_step 10000 \
    --step_size 1000 \
    --batch_size 8 \
    --max_len 2048 \
    --compute_seq_metrics
"""

import os
import gc
import argparse
import time
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from gLM.tokenizers import PhyloTokenizerLoader
from gLM.models.protein_modernbert_phylo import ProteinModernBertPrefixLM
from gLM.attention_mask.prefixlm_flash import run_encoder_flash

MODEL_VOCAB_SIZE = 27


# =============================================================================
# Vocab
# =============================================================================

def build_vocab(tokenizer):
    return {
        token_id: token_name
        for token_name, token_id in tokenizer.get_vocab().items()
        if token_id < MODEL_VOCAB_SIZE
    }


# =============================================================================
# CORE COMPUTE
# =============================================================================

def compute_metrics_prefixlm(
    model,
    tokenizer,
    seqs,
    poses,
    refs,
    alts,
    max_len,
    device,
    vocab,
    compute_seq_metrics=False,
):
    results = [None] * len(seqs)

    valid_data = []

    for i, (seq, pos, ref, alt) in enumerate(zip(seqs, poses, refs, alts)):
        pos = int(pos)

        if len(seq) > max_len or pos >= len(seq) or seq[pos] != ref:
            continue

        ref_id = tokenizer.convert_tokens_to_ids(ref)
        alt_id = tokenizer.convert_tokens_to_ids(alt)

        if ref_id is None or alt_id is None:
            continue

        valid_data.append((i, seq, pos, ref_id, alt_id))

    if not valid_data:
        return results

    indices, valid_seqs, valid_poses, ref_ids, alt_ids = zip(*valid_data)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    all_input_ids = []
    all_prefix_lengths = []
    all_logit_positions = []

    # Important:
    # Store metadata only for examples that survive the packed-length filter.
    # Otherwise indices/ref_ids/alt_ids become misaligned after skipped examples.
    surviving = []

    for orig_idx, seq, pos, ref_id, alt_id in zip(
        indices, valid_seqs, valid_poses, ref_ids, alt_ids
    ):
        enc = tokenizer(seq, add_special_tokens=False)["input_ids"]

        # Pack: [CLS] wildtype [SEP] wildtype[:pos] [SEP]
        packed = [cls_id] + enc + [sep_id] + enc[:pos] + [sep_id]
        prefix_len = 1 + len(enc) + 1

        if len(packed) > max_len:
            continue

        # Last token is [SEP], second-to-last token predicts wildtype[pos]
        logit_pos = len(packed) - 2

        all_input_ids.append(packed)
        all_prefix_lengths.append(prefix_len)
        all_logit_positions.append(logit_pos)

        surviving.append(
            {
                "orig_idx": orig_idx,
                "ref_id": ref_id,
                "alt_id": alt_id,
                "seq": seq,
                "pos": pos,
            }
        )

    if not all_input_ids:
        return results

    max_len_batch = max(len(x) for x in all_input_ids)
    padded = [x + [pad_id] * (max_len_batch - len(x)) for x in all_input_ids]

    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    prefix_lengths = torch.tensor(all_prefix_lengths, dtype=torch.long, device=device)

    base_model = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = run_encoder_flash(model, input_ids, prefix_lengths, device)
            logits = base_model.decoder(base_model.head(hidden))
            logits = logits[:, :-1, :].contiguous()

    for b in range(len(all_input_ids)):
        info = surviving[b]

        orig_idx = info["orig_idx"]
        ref_id = info["ref_id"]
        alt_id = info["alt_id"]
        seq = info["seq"]
        pos = info["pos"]

        lpos = all_logit_positions[b]

        if lpos >= logits.shape[1]:
            continue

        logit = logits[b, lpos, :]
        log_probs = torch.nn.functional.log_softmax(logit.float(), dim=-1)
        probs = torch.exp(log_probs)

        entropy = -(probs * log_probs).sum().item()
        log_odds = (log_probs[alt_id] - log_probs[ref_id]).item()

        result = {
            "log_odds": log_odds,
            "model_score": -log_odds,
            "ref_log_prob": log_probs[ref_id].item(),
            "alt_log_prob": log_probs[alt_id].item(),
            "entropy": entropy,
        }

        # Full distribution
        probs_np = probs.detach().cpu().numpy()

        for token_id, token_name in vocab.items():
            result[f"prob_{token_name}"] = float(probs_np[token_id])

        # OPTIONAL sequence metrics
        if compute_seq_metrics:
            seq_log_prob = 0.0
            seq_entropy = 0.0
            count = 0

            prefix_len = all_prefix_lengths[b]

            # In this PrefixLM setup, suffix is seq[:pos], not the full sequence.
            # So only compute metrics over available predicted suffix tokens.
            suffix_len = pos

            for pos_i in range(prefix_len, prefix_len + suffix_len):
                if pos_i >= logits.shape[1]:
                    continue

                lp = torch.nn.functional.log_softmax(
                    logits[b, pos_i, :].float(), dim=-1
                )
                prob = torch.exp(lp)

                true_id = input_ids[b, pos_i + 1]

                seq_log_prob += lp[true_id].item()
                seq_entropy += -(prob * lp).sum().item()
                count += 1

            if count > 0:
                seq_log_prob /= count
                seq_entropy /= count

            result["seq_log_prob"] = seq_log_prob
            result["seq_entropy"] = seq_entropy

        results[orig_idx] = result

    return results


# =============================================================================
# CHECKPOINT RUNNER
# =============================================================================

def run_checkpoint(model, tokenizer, df, args, device, vocab, step):
    n = len(df)
    scores = [None] * n

    checkpoint_start = time.time()

    print(
        f"[step {step}] Starting scoring for {n:,} variants "
        f"with batch_size={args.batch_size}",
        flush=True,
    )

    for i in range(0, n, args.batch_size):
        batch_start = time.time()

        batch_scores = compute_metrics_prefixlm(
            model=model,
            tokenizer=tokenizer,
            seqs=df["sequence"].values[i : i + args.batch_size],
            poses=df["pos"].values[i : i + args.batch_size],
            refs=df["ref"].values[i : i + args.batch_size],
            alts=df["alt"].values[i : i + args.batch_size],
            max_len=args.max_len,
            device=device,
            vocab=vocab,
            compute_seq_metrics=args.compute_seq_metrics,
        )

        n_scored_batch = 0

        for j, result in enumerate(batch_scores):
            if result is not None:
                scores[i + j] = result
                n_scored_batch += 1

        # Print progress every 100 batches, plus first and last batch
        batch_num = (i // args.batch_size) + 1
        total_batches = (n + args.batch_size - 1) // args.batch_size

        if batch_num == 1 or batch_num % 100 == 0 or batch_num == total_batches:
            elapsed = time.time() - checkpoint_start
            scored_so_far = sum(s is not None for s in scores)

            print(
                f"[step {step}] Batch {batch_num:,}/{total_batches:,} | "
                f"rows {i:,}-{min(i + args.batch_size, n):,} | "
                f"scored in batch={n_scored_batch} | "
                f"scored so far={scored_so_far:,} | "
                f"elapsed={elapsed/60:.2f} min | "
                f"last_batch_time={time.time() - batch_start:.2f}s",
                flush=True,
            )

    total_time = time.time() - checkpoint_start
    total_scored = sum(s is not None for s in scores)

    print(
        f"[step {step}] Finished scoring. "
        f"Total scored={total_scored:,}/{n:,}. "
        f"Time={total_time/60:.2f} min",
        flush=True,
    )

    return scores


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_checkpoint(path, tokenizer, device):
    print(f"Loading model from: {path}", flush=True)

    model = ProteinModernBertPrefixLM(
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer,
    ).build()

    safetensor_path = os.path.join(path, "model.safetensors")
    pytorch_path = os.path.join(path, "pytorch_model.bin")

    if os.path.exists(safetensor_path):
        print(f"Loading safetensors checkpoint: {safetensor_path}", flush=True)
        state = load_file(safetensor_path, device=str(device))
    else:
        print(f"Loading PyTorch checkpoint: {pytorch_path}", flush=True)
        state = torch.load(pytorch_path, map_location=device)

    missing, unexpected = model.load_state_dict(state, strict=False)

    print(
        f"Loaded checkpoint. Missing keys={len(missing)}, "
        f"unexpected keys={len(unexpected)}",
        flush=True,
    )

    model.to(device)
    model.eval()

    print("Model moved to device and set to eval mode.", flush=True)

    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--vep_csv", required=True)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--start_step", type=int, required=True)
    parser.add_argument("--end_step", type=int, required=True)
    parser.add_argument("--step_size", type=int, required=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=2048)

    parser.add_argument("--compute_seq_metrics", action="store_true")

    args = parser.parse_args()

    print("=" * 80, flush=True)
    print("Starting ClinVar PrefixLM scoring script", flush=True)
    print("=" * 80, flush=True)

    print(f"checkpoint_dir: {args.checkpoint_dir}", flush=True)
    print(f"tokenizer_path:  {args.tokenizer_path}", flush=True)
    print(f"vep_csv:         {args.vep_csv}", flush=True)
    print(f"output_csv:      {args.output_csv}", flush=True)
    print(f"start_step:      {args.start_step}", flush=True)
    print(f"end_step:        {args.end_step}", flush=True)
    print(f"step_size:       {args.step_size}", flush=True)
    print(f"batch_size:      {args.batch_size}", flush=True)
    print(f"max_len:         {args.max_len}", flush=True)
    print(f"seq metrics:     {args.compute_seq_metrics}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}", flush=True)

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
        print(
            f"GPU memory allocated at start: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            flush=True,
        )

    print("Loading tokenizer...", flush=True)
    tokenizer = PhyloTokenizerLoader(args.tokenizer_path)
    print("Tokenizer loaded.", flush=True)

    print("Building vocab...", flush=True)
    vocab = build_vocab(tokenizer)
    print(f"Vocab size used for probabilities: {len(vocab)}", flush=True)

    print("Reading ClinVar CSV...", flush=True)
    df = pd.read_csv(args.vep_csv)
    print(f"Loaded {len(df):,} variants from ClinVar CSV.", flush=True)
    print(f"Input columns: {list(df.columns)}", flush=True)

    # Preserve original ClinVar row identity.
    # Do not reset index after filtering.
    df = df.reset_index(drop=False).rename(columns={"index": "original_clinvar_idx"})

    print("Label counts in original input:", flush=True)
    print(df["label"].value_counts(dropna=False), flush=True)

    steps = list(range(args.start_step, args.end_step + 1, args.step_size))
    print(f"Requested steps: {steps}", flush=True)

    all_rows = []

    script_start = time.time()

    for step in steps:
        path = os.path.join(args.checkpoint_dir, f"checkpoint-{step}")

        print("\n" + "=" * 80, flush=True)
        print(f"Preparing checkpoint step {step}", flush=True)
        print("=" * 80, flush=True)

        if not os.path.exists(path):
            print(f"Skipping checkpoint {step}: {path} does not exist", flush=True)
            continue

        step_start = time.time()

        model = load_checkpoint(path, tokenizer, device)

        if torch.cuda.is_available():
            print(
                f"[step {step}] GPU memory after model load: "
                f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                flush=True,
            )

        scores = run_checkpoint(
            model=model,
            tokenizer=tokenizer,
            df=df,
            args=args,
            device=device,
            vocab=vocab,
            step=step,
        )

        print(f"[step {step}] Converting scores to rows...", flush=True)

        step_rows = []

        for i, s in enumerate(scores):
            row = {
                "step": step,
                "variant_idx": int(df["original_clinvar_idx"].iloc[i]),
                "pos": int(df["pos"].iloc[i]),
                "ref": df["ref"].iloc[i],
                "alt": df["alt"].iloc[i],
                "label": int(df["label"].iloc[i]),
                "scored": s is not None,
            }

            if s is not None:
                row.update(s)

            step_rows.append(row)

        all_rows.extend(step_rows)

        step_df = pd.DataFrame(step_rows)
        scored_step_df = step_df[step_df["scored"]]

        print(f"[step {step}] Rows added: {len(step_rows):,}", flush=True)
        print(f"[step {step}] Scored variants: {len(scored_step_df):,}", flush=True)

        print(f"[step {step}] Scored label counts:", flush=True)
        print(scored_step_df["label"].value_counts().sort_index(), flush=True)

        print(
            f"[step {step}] Finished checkpoint in "
            f"{(time.time() - step_start) / 60:.2f} min",
            flush=True,
        )

        del model
        torch.cuda.empty_cache()
        gc.collect()

        if torch.cuda.is_available():
            print(
                f"[step {step}] GPU memory after cleanup: "
                f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                flush=True,
            )

    print("\n" + "=" * 80, flush=True)
    print("Writing final output CSV...", flush=True)
    print("=" * 80, flush=True)

    out = pd.DataFrame(all_rows)

    print(f"Total output rows: {len(out):,}", flush=True)

    out.to_csv(args.output_csv, index=False)

    print("\nDone!", flush=True)
    print(f"Wrote {len(out):,} rows to {args.output_csv}", flush=True)

    if len(out) > 0:
        scored = out[out["scored"]]

        print("\nNumber of scored variants by step:", flush=True)
        print(scored.groupby("step").size(), flush=True)

        print("\nScored variants per label by step:", flush=True)
        print(scored.groupby(["step", "label"]).size(), flush=True)

    print(
        f"\nTotal script time: {(time.time() - script_start) / 60:.2f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()