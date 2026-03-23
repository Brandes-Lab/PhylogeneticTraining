#!/usr/bin/env python3
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import roc_auc_score

from transformers import (
    HfArgumentParser,
    PreTrainedTokenizerFast,
    T5GemmaForConditionalGeneration,
)

# -----------------------------
# DDP helpers
# -----------------------------
def is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_initialized() else 1

def all_gather_object(gathered, local):
    if is_initialized():
        dist.all_gather_object(gathered, local)
    else:
        gathered[0] = local

def maybe_init_distributed():
    if dist.is_available() and "RANK" in os.environ and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

# -----------------------------
# Args
# -----------------------------
@dataclass
class EvalArguments:
    model_ckpt: str = field(metadata={"help": "HF checkpoint repo id or local folder"})
    zero_shot_csv: str = field(metadata={"help": "CSV with columns: sequence,pos,ref,alt,label"})
    max_len: int = field(metadata={"help": "Tokenizer truncation cap"})
    batch_size: int = field(default=8)
    device: Optional[str] = field(default=None)
    run_name: str = field(default="pll_vep_eval")
    step_id: int = field(default=0)
    local_rank: int = field(default=-1)

    pll_mode: str = field(
        default="wtenc",
        metadata={"help": "PLL mode: wtenc | selfenc | singlepos | both | all"},
    )

    out_dir: str = field(
        default=".",
        metadata={"help": "Directory to write the AUC CSV (rank 0 only)."},
    )

# -----------------------------
# PLL utilities
# -----------------------------
def shift_right(input_ids: torch.Tensor, start_id: int, pad_id: int) -> torch.Tensor:
    """
    Teacher-forcing shift for seq2seq decoders.

    input_ids: LongTensor [B, T] (target token IDs, padded)
    returns:   LongTensor [B, T]
      out[:,0] = start_id
      out[:,1:] = input_ids[:,:-1]
    """
    B, T = input_ids.shape
    shifted = input_ids.new_full((B, T), fill_value=pad_id)
    shifted[:, 0] = start_id
    shifted[:, 1:] = input_ids[:, :-1]
    return shifted


@torch.no_grad()
def pll_batch_seq2seq_conditional(
    model,
    tokenizer,
    encoder_seqs: List[str],
    target_seqs: List[str],
    max_len: int,
) -> torch.Tensor:
    """
    Compute PLL(target | encoder) in batch:

      PLL = sum_t log P(target_t | target_<t, encoder(encoder_seq))

    encoder_seqs: list[str] length B  (what encoder sees)
    target_seqs:  list[str] length B  (what decoder is scored on)
    returns: FloatTensor [B]
    """
    assert len(encoder_seqs) == len(target_seqs), "encoder_seqs and target_seqs must match length"
    device = next(model.parameters()).device

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id set; required for padding/PLL.")

    pad_id = tokenizer.pad_token_id
    decoder_start_id = getattr(model.config, "decoder_start_token_id", None)
    if decoder_start_id is None:
        decoder_start_id = pad_id

    # --- tokenize encoder ---
    enc = tokenizer(
        encoder_seqs,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    ).to(device)
    enc_input_ids = enc["input_ids"]
    enc_attention_mask = enc["attention_mask"]

    # --- tokenize decoder/targets ---
    dec = tokenizer(
        target_seqs,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    ).to(device)
    tgt_input_ids = dec["input_ids"]
    tgt_attention_mask = dec["attention_mask"]

    labels = tgt_input_ids.clone()
    labels = labels.masked_fill(tgt_attention_mask == 0, -100)

    decoder_input_ids = shift_right(tgt_input_ids, decoder_start_id, pad_id)
    decoder_attention_mask = shift_right(tgt_attention_mask, 1, 0)

    outputs = model(
        input_ids=enc_input_ids,
        attention_mask=enc_attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
    )

    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)

    gather_labels = labels.clone()
    gather_labels[gather_labels == -100] = 0
    token_logp = log_probs.gather(
        dim=-1, index=gather_labels.unsqueeze(-1)
    ).squeeze(-1)

    token_logp = token_logp * (labels != -100).to(token_logp.dtype)
    return token_logp.sum(dim=1)


@torch.no_grad()
def singlepos_batch_seq2seq(
    model,
    tokenizer,
    wt_seqs: List[str],
    positions: List[int],
    refs: List[str],
    alts: List[str],
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single-position scoring for encoder-decoder VEP.

    For each variant, we:
      1. Encode the WT sequence with the encoder
      2. Feed the WT sequence as the decoder target (teacher-forced)
      3. At the mutation position, read off log P(ref) and log P(alt)

    This avoids full-sequence PLL and directly probes the model's
    residue-level preference at the variant site.

    Returns:
        ref_logp: FloatTensor [B] - log P(ref_token | wt_<pos, enc(wt))
        alt_logp: FloatTensor [B] - log P(alt_token | wt_<pos, enc(wt))
    """
    device = next(model.parameters()).device

    pad_id = tokenizer.pad_token_id
    decoder_start_id = getattr(model.config, "decoder_start_token_id", None)
    if decoder_start_id is None:
        decoder_start_id = pad_id

    # --- tokenize encoder (WT sequences) ---
    enc = tokenizer(
        wt_seqs,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    ).to(device)
    enc_input_ids = enc["input_ids"]
    enc_attention_mask = enc["attention_mask"]

    # --- tokenize decoder target (WT sequences, teacher-forced) ---
    # We use WT as the decoder target so the decoder prefix up to `pos`
    # is conditioned on the true WT residues. At position `pos` in the
    # decoder output, the model has seen wt_<pos as input and we read
    # off P(ref) vs P(alt).
    dec = tokenizer(
        wt_seqs,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    ).to(device)
    tgt_input_ids = dec["input_ids"]       # [B, T]
    tgt_attention_mask = dec["attention_mask"]

    decoder_input_ids = shift_right(tgt_input_ids, decoder_start_id, pad_id)
    decoder_attention_mask = shift_right(tgt_attention_mask, 1, 0)

    outputs = model(
        input_ids=enc_input_ids,
        attention_mask=enc_attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
    )

    logits = outputs.logits                    # [B, T, V]
    log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]

    B = len(wt_seqs)
    ref_logp = torch.zeros(B, device=device)
    alt_logp = torch.zeros(B, device=device)

    for i in range(B):
        pos = positions[i]

        # The decoder output at index `pos` corresponds to predicting
        # the token at position `pos` in the target, given:
        #   - encoder hidden states from enc(wt)
        #   - decoder input tokens [start, tgt_0, tgt_1, ..., tgt_{pos-1}]
        #
        # Because of shift_right, decoder_input_ids[:, pos] = tgt_input_ids[:, pos-1]
        # and logits[:, pos, :] predicts tgt_input_ids[:, pos].
        #
        # So logits[i, pos, :] gives P(token | wt_<pos, enc(wt)) which is
        # exactly what we want.

        # Handle the offset: with add_special_tokens=False, token index
        # in tgt_input_ids should align 1:1 with sequence position.
        # But we verify via the ref token.
        ref_token_id = tokenizer.convert_tokens_to_ids(refs[i])
        alt_token_id = tokenizer.convert_tokens_to_ids(alts[i])

        ref_logp[i] = log_probs[i, pos, ref_token_id]
        alt_logp[i] = log_probs[i, pos, alt_token_id]

    return ref_logp, alt_logp


def compute_pll_delta(
    model,
    tokenizer,
    seqs: List[str],
    poses: np.ndarray,
    refs: List[str],
    alts: List[str],
    max_len: int,
    pll_mode: str,
) -> List[Optional[float]]:
    """
    Returns: list[float|None] aligned to input order

    wtenc:
      Delta = PLL(mut | enc(wt))  - PLL(wt | enc(wt))

    selfenc:
      Delta = PLL(mut | enc(mut)) - PLL(wt | enc(wt))

    singlepos:
      Delta = log P(alt | wt_<pos, enc(wt)) - log P(ref | wt_<pos, enc(wt))
      Only scores the mutation position — no full-sequence PLL.
    """
    if pll_mode not in ("wtenc", "selfenc", "singlepos"):
        raise ValueError(f"pll_mode must be 'wtenc', 'selfenc', or 'singlepos', got {pll_mode}")

    results: List[Optional[float]] = [None] * len(seqs)
    valid: List[Tuple[int, str, int, str, str]] = []

    for i, (seq, pos, ref, alt) in enumerate(zip(seqs, poses, refs, alts)):
        if len(seq) <= max_len and 0 <= int(pos) < len(seq) and seq[int(pos)] == ref:
            ref_id = tokenizer.convert_tokens_to_ids(ref)
            alt_id = tokenizer.convert_tokens_to_ids(alt)
            if ref_id is None or alt_id is None:
                continue
            valid.append((i, seq, int(pos), ref, alt))

    if not valid:
        return results

    indices = [x[0] for x in valid]
    wt_seqs  = [x[1] for x in valid]
    poses_v  = [x[2] for x in valid]
    refs_v   = [x[3] for x in valid]
    alts_v   = [x[4] for x in valid]

    if pll_mode == "singlepos":
        ref_logp, alt_logp = singlepos_batch_seq2seq(
            model=model,
            tokenizer=tokenizer,
            wt_seqs=wt_seqs,
            positions=poses_v,
            refs=refs_v,
            alts=alts_v,
            max_len=max_len,
        )
        # Delta = log P(alt | context) - log P(ref | context)
        delta = (alt_logp - ref_logp).tolist()
        for idx, d in zip(indices, delta):
            results[idx] = float(d)
        return results

    # --- original full-sequence PLL modes ---
    mut_seqs = [wt[:pos] + alt + wt[pos + 1:] for wt, pos, alt in zip(wt_seqs, poses_v, alts_v)]

    wt_pll = pll_batch_seq2seq_conditional(
        model, tokenizer,
        encoder_seqs=wt_seqs,
        target_seqs=wt_seqs,
        max_len=max_len
    )

    if pll_mode == "wtenc":
        mut_pll = pll_batch_seq2seq_conditional(
            model, tokenizer,
            encoder_seqs=wt_seqs,
            target_seqs=mut_seqs,
            max_len=max_len
        )
    else:  # selfenc
        mut_pll = pll_batch_seq2seq_conditional(
            model, tokenizer,
            encoder_seqs=mut_seqs,
            target_seqs=mut_seqs,
            max_len=max_len
        )

    delta = (mut_pll - wt_pll).tolist()
    for idx, d in zip(indices, delta):
        results[idx] = float(d)
    return results


def _safe_name_from_ckpt_path(model_ckpt: str) -> Tuple[str, str, str]:
    p = model_ckpt.rstrip("/")
    if os.path.exists(p):
        ckpt_name = os.path.basename(p)
        parent = os.path.basename(os.path.dirname(p)) or ckpt_name
        model_dir_name = parent
        checkpoint_dir_name = ckpt_name
    else:
        model_dir_name = p.replace("/", "__")
        checkpoint_dir_name = model_dir_name

    csv_basename = f"{model_dir_name}__{checkpoint_dir_name}__pll_vep_auc.csv"
    return model_dir_name, checkpoint_dir_name, csv_basename


def run_vep_eval(
    df: pd.DataFrame,
    model,
    tokenizer,
    batch_size: int,
    max_len: int,
    step_id: int,
    pll_mode: str,
) -> Dict[str, object]:
    rank = get_rank()
    world_size = get_world_size()

    # Expand "both" and "all" into the list of modes to run
    if pll_mode == "both":
        modes_to_run = ["wtenc", "selfenc"]
    elif pll_mode == "all":
        modes_to_run = ["wtenc", "selfenc", "singlepos"]
    elif pll_mode in ("wtenc", "selfenc", "singlepos"):
        modes_to_run = [pll_mode]
    else:
        raise ValueError(f"--pll_mode must be wtenc/selfenc/singlepos/both/all, got {pll_mode}")

    print(f"[Rank {rank}] Starting PLL VEP eval @ step {step_id} (modes={modes_to_run})", flush=True)

    seqs = df["sequence"].tolist()
    poses = df["pos"].to_numpy(dtype=np.int64)
    refs = df["ref"].tolist()
    alts = df["alt"].tolist()
    labels = df["label"].to_numpy(dtype=np.int8)

    n = len(labels)
    indices = np.arange(n)
    shard_indices = indices[rank::world_size]

    # Allocate prediction arrays per mode
    preds_per_mode: Dict[str, np.ndarray] = {}
    for mode in modes_to_run:
        preds_per_mode[mode] = np.full(len(shard_indices), np.nan, dtype=np.float32)

    was_training = model.training
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, len(shard_indices), batch_size):
            batch_ids = shard_indices[i:i + batch_size]

            batch_seqs = [seqs[k] for k in batch_ids]
            batch_poses = poses[batch_ids]
            batch_refs = [refs[k] for k in batch_ids]
            batch_alts = [alts[k] for k in batch_ids]

            for mode in modes_to_run:
                deltas = compute_pll_delta(
                    model=model, tokenizer=tokenizer,
                    seqs=batch_seqs, poses=batch_poses,
                    refs=batch_refs, alts=batch_alts,
                    max_len=max_len, pll_mode=mode,
                )
                for j, d in enumerate(deltas):
                    if d is not None:
                        # For wtenc/selfenc: negate so higher = more pathogenic
                        # For singlepos: delta is log P(alt) - log P(ref),
                        #   negate so that destabilizing mutations (low P(alt))
                        #   get higher scores = more pathogenic
                        preds_per_mode[mode][i + j] = -float(d)

            if (i % 20) == 0:
                print(f"[Rank {rank}] Progress: {i}/{len(shard_indices)}", flush=True)

    if was_training:
        model.train()

    # Gather across ranks
    # Pack: (global_idx, label, pred_mode1, pred_mode2, ...)
    gathered_data = [None for _ in range(world_size)]
    local_data = []
    for local_i, global_idx in enumerate(shard_indices.tolist()):
        row = [global_idx, int(labels[global_idx])]
        for mode in modes_to_run:
            row.append(float(preds_per_mode[mode][local_i]))
        local_data.append(tuple(row))
    all_gather_object(gathered_data, local_data)

    results: Dict[str, object] = {
        "step_id": int(step_id),
        "pll_mode": pll_mode,
        "n_total": int(n),
        "elapsed_seconds": float(time.time() - start_time),
    }
    # Initialize AUC fields
    for mode in ["wtenc", "selfenc", "singlepos"]:
        results[f"auc_pll_{mode}"] = np.nan
        results[f"n_scored_{mode}"] = 0

    if rank == 0:
        flat_preds: Dict[str, np.ndarray] = {}
        for mode in modes_to_run:
            flat_preds[mode] = np.full(n, np.nan, dtype=np.float32)

        for part in gathered_data:
            for row in part:
                idx = row[0]
                for mi, mode in enumerate(modes_to_run):
                    flat_preds[mode][idx] = row[2 + mi]

        def compute_auc(preds: np.ndarray) -> Tuple[float, int]:
            mask = ~np.isnan(preds)
            n_scored = int(mask.sum())
            if n_scored >= 10 and (labels[mask].min() != labels[mask].max()):
                return float(roc_auc_score(labels[mask], preds[mask])), n_scored
            return float("nan"), n_scored

        for mode in modes_to_run:
            auc, n_scored = compute_auc(flat_preds[mode])
            results[f"auc_pll_{mode}"] = auc
            results[f"n_scored_{mode}"] = n_scored
            if not np.isnan(auc):
                print(f"AUC (PLL {mode}) at step {step_id}: {auc:.4f}", flush=True)
            else:
                print(f"Skipping AUC (PLL {mode}) due to insufficient data", flush=True)

        print(f"[TIMER] VEP eval took {results['elapsed_seconds']:.2f} seconds", flush=True)

    return results


def main():
    parser = HfArgumentParser((EvalArguments,))
    (eval_args,) = parser.parse_args_into_dataclasses()

    maybe_init_distributed()
    rank = get_rank()

    if eval_args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        eval_args.local_rank = int(os.environ["LOCAL_RANK"])

    if eval_args.device is not None:
        device = torch.device(eval_args.device)
    else:
        if torch.cuda.is_available():
            if eval_args.local_rank not in (-1, None):
                torch.cuda.set_device(eval_args.local_rank)
                device = torch.device(f"cuda:{eval_args.local_rank}")
            else:
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if rank == 0:
        print(f"model_ckpt: {eval_args.model_ckpt}", flush=True)
        print(f"zero_shot_csv: {eval_args.zero_shot_csv}", flush=True)
        print(f"max_len: {eval_args.max_len}", flush=True)
        print(f"batch_size: {eval_args.batch_size}", flush=True)
        print(f"run_name: {eval_args.run_name}", flush=True)
        print(f"pll_mode: {eval_args.pll_mode}", flush=True)
        print(f"out_dir: {eval_args.out_dir}", flush=True)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(eval_args.model_ckpt)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id. Set/define it before PLL evaluation.")

    model = T5GemmaForConditionalGeneration.from_pretrained(eval_args.model_ckpt)
    model = model.to(device).eval()

    df = pd.read_csv(eval_args.zero_shot_csv)
    required = {"sequence", "pos", "ref", "alt", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df["pos"] = df["pos"].astype(int)

    results = run_vep_eval(
        df=df,
        model=model,
        tokenizer=tokenizer,
        batch_size=eval_args.batch_size,
        max_len=eval_args.max_len,
        step_id=eval_args.step_id,
        pll_mode=eval_args.pll_mode,
    )

    if rank == 0:
        model_dir_name, checkpoint_dir_name, csv_basename = _safe_name_from_ckpt_path(eval_args.model_ckpt)
        os.makedirs(eval_args.out_dir, exist_ok=True)
        out_path = os.path.join(eval_args.out_dir, csv_basename)

        row = {
            "model_ckpt": eval_args.model_ckpt,
            "model_dir": model_dir_name,
            "checkpoint_dir": checkpoint_dir_name,
            "zero_shot_csv": eval_args.zero_shot_csv,
            **results,
        }
        pd.DataFrame([row]).to_csv(out_path, index=False)
        print(f"[Rank 0] Wrote AUC results to: {out_path}", flush=True)


if __name__ == "__main__":
    main()