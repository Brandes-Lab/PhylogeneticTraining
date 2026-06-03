#!/usr/bin/env python3
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

REPO_ROOT = "/gpfs/data/brandeslab/User/as12267/HuggingfaceTransformer"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from transformers import (
    HfArgumentParser,
    PreTrainedTokenizerFast,
    ModernBertConfig,
    ModernBertForMaskedLM,
)

from gLM.attention_mask import run_encoder_flash

# -----------------------------
# Args
# -----------------------------
@dataclass
class EvalArguments:
    model_ckpt: str = field(metadata={"help": "Local checkpoint folder"})
    zero_shot_csv: str = field(metadata={"help": "CSV with columns: sequence,pos,ref,alt,label"})
    max_len: int = field(default=1024)
    batch_size: int = field(default=8)
    device: Optional[str] = field(default=None)
    run_name: str = field(default="prefixlm_full_ll_vep_eval")
    step_id: int = field(default=0)

    out_dir: str = field(
        default=".",
        metadata={"help": "Directory to write AUC CSV."},
    )
    score_type: str = field(
        default="full_ll",
        metadata={"help": "Scoring mode: full_ll or single_pos"},
    )
    per_variant_csv: bool = field(
        default=False,
        metadata={
            "help": (
                "If set AND score_type == 'single_pos', also write a per-variant CSV "
                "with columns: step, variant_idx, pos, ref, alt, label, scored, "
                "ref_log_prob, alt_log_prob, log_odds, model_score. "
                "Ignored when score_type == 'full_ll'."
            )
        },
    )


# -----------------------------
# Utilities
# -----------------------------
def _safe_name_from_ckpt_path(model_ckpt: str, score_type: str) -> Tuple[str, str, str]:
    """
    Example:
        .../<model_dir>/checkpoint-15060

    returns:
        model_dir, checkpoint-15060, model_dir__checkpoint-15060__prefixlm_full_ll_auc.csv
    """
    p = model_ckpt.rstrip("/")

    if os.path.exists(p):
        ckpt_name = os.path.basename(p)
        parent = os.path.basename(os.path.dirname(p)) or ckpt_name
        model_dir_name = parent
        checkpoint_dir_name = ckpt_name
    else:
        model_dir_name = p.replace("/", "__")
        checkpoint_dir_name = model_dir_name

    csv_basename = f"{model_dir_name}__{checkpoint_dir_name}__prefixlm_full_ll_auc.csv"
    return model_dir_name, checkpoint_dir_name, csv_basename


def _per_variant_csv_basename(model_ckpt: str) -> str:
    """Filename for the single-position per-variant CSV."""
    p = model_ckpt.rstrip("/")
    if os.path.exists(p):
        ckpt_name = os.path.basename(p)
        parent = os.path.basename(os.path.dirname(p)) or ckpt_name
        model_dir_name = parent
        checkpoint_dir_name = ckpt_name
    else:
        model_dir_name = p.replace("/", "__")
        checkpoint_dir_name = model_dir_name
    return f"{model_dir_name}__{checkpoint_dir_name}__prefixlm_single_pos_per_variant.csv"


def load_prefixlm_model(model_ckpt: str, tokenizer, device: torch.device):
    """
    Load the trained PrefixLM checkpoint.

    ProteinModernBertPrefixLM is only your builder class.
    The actual saved model is ModernBertForMaskedLM.
    """
    config = ModernBertConfig.from_pretrained(model_ckpt)
    config._attn_implementation = "flash_attention_2"

    # Keep tokenizer IDs consistent with checkpoint/eval tokenizer.
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id

    model = ModernBertForMaskedLM.from_pretrained(
        model_ckpt,
        config=config,
        torch_dtype=torch.bfloat16,
    )

    model = model.to(device)
    model.eval()
    return model


# -----------------------------
# PrefixLM full LL scoring  (UNCHANGED)
# -----------------------------
@torch.no_grad()
def compute_log_odds_prefixlm_full_LL(
    model,
    tokenizer,
    seqs: List[str],
    poses: np.ndarray,
    refs: List[str],
    alts: List[str],
    max_len: int,
) -> List[Optional[float]]:
    """
    Computes PrefixLM full log-likelihood variant score.

    For each variant:

        score = LL(ALT from mutation position onward)
              - LL(REF from mutation position onward)

    Packing:

        REF row:
            [CLS] wildtype_full [SEP] wildtype_full [SEP]

        ALT row:
            [CLS] wildtype_full [SEP] mutant_full [SEP]

    The prefix/context is always wildtype_full.

    Only candidate suffix amino acid tokens from `pos` onward are included
    in the LL sum. Special tokens, prefix tokens, padding, and unchanged
    suffix tokens before `pos` are not scored.

    Returns:
        list of scores aligned to input batch order.
        None means invalid or skipped.
    """

    results: List[Optional[float]] = [None] * len(seqs)
    device = next(model.parameters()).device

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    if cls_id is None:
        raise ValueError("tokenizer.cls_token_id is None.")
    if sep_id is None:
        raise ValueError("tokenizer.sep_token_id is None.")
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id is None.")

    valid_data = []

    for i, (seq, pos, ref, alt) in enumerate(zip(seqs, poses, refs, alts)):
        pos = int(pos)

        if pos < 0 or pos >= len(seq):
            continue
        if seq[pos] != ref:
            continue

        ref_id = tokenizer.convert_tokens_to_ids(ref)
        alt_id = tokenizer.convert_tokens_to_ids(alt)

        if ref_id is None or alt_id is None:
            continue

        alt_seq = seq[:pos] + alt + seq[pos + 1:]
        valid_data.append((i, seq, alt_seq, pos))

    if not valid_data:
        return results

    all_input_ids = []
    all_prefix_lengths = []
    all_score_start_positions = []
    row_to_orig = []
    row_is_alt = []

    for orig_idx, ref_seq, alt_seq, pos in valid_data:
        enc_ref_full = tokenizer(ref_seq, add_special_tokens=False)["input_ids"]
        enc_alt_full = tokenizer(alt_seq, add_special_tokens=False)["input_ids"]

        if len(enc_ref_full) != len(ref_seq):
            continue
        if len(enc_alt_full) != len(alt_seq):
            continue

        prefix = enc_ref_full
        prefix_len = 1 + len(prefix) + 1  # [CLS] + WT prefix + [SEP]

        score_start = prefix_len + pos - 1

        for candidate_ids, is_alt in ((enc_ref_full, False), (enc_alt_full, True)):
            packed = [cls_id] + prefix + [sep_id] + candidate_ids + [sep_id]

            if len(packed) > max_len:
                continue

            all_input_ids.append(packed)
            all_prefix_lengths.append(prefix_len)
            all_score_start_positions.append(score_start)
            row_to_orig.append(orig_idx)
            row_is_alt.append(is_alt)

    if not all_input_ids:
        return results

    max_batch_len = max(len(ids) for ids in all_input_ids)

    padded_input_ids = [
        ids + [pad_id] * (max_batch_len - len(ids))
        for ids in all_input_ids
    ]

    input_ids_tensor = torch.tensor(
        padded_input_ids,
        dtype=torch.long,
        device=device,
    )

    prefix_lengths_tensor = torch.tensor(
        all_prefix_lengths,
        dtype=torch.long,
        device=device,
    )

    score_start_tensor = torch.tensor(
        all_score_start_positions,
        dtype=torch.long,
        device=device,
    )

    row_to_orig = np.array(row_to_orig)
    row_is_alt = np.array(row_is_alt, dtype=bool)

    base_model = model.module if hasattr(model, "module") else model

    if device.type == "cuda":
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = torch.autocast(device_type="cpu", enabled=False)

    with autocast_context:
        hidden_states = run_encoder_flash(
            model,
            input_ids_tensor,
            prefix_lengths_tensor,
            device,
        )

        logits = base_model.decoder(base_model.head(hidden_states))

    logits_shifted = logits[:, :-1, :].contiguous()
    target_ids = input_ids_tensor[:, 1:].contiguous()

    log_probs = F.log_softmax(logits_shifted.float(), dim=-1)

    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    _, shifted_len = target_ids.shape
    positions = torch.arange(shifted_len, device=device).unsqueeze(0)

    seq_lens = (input_ids_tensor != pad_id).sum(dim=1)
    score_end_tensor = seq_lens - 2

    score_mask = (
        (positions >= score_start_tensor.unsqueeze(1))
        & (positions < score_end_tensor.unsqueeze(1))
    )

    ll_per_row = (token_log_probs * score_mask.to(token_log_probs.dtype)).sum(dim=1)

    ll_ref_by_orig: Dict[int, float] = {}
    ll_alt_by_orig: Dict[int, float] = {}

    for row_idx, orig_idx in enumerate(row_to_orig):
        ll_value = float(ll_per_row[row_idx].item())

        if row_is_alt[row_idx]:
            ll_alt_by_orig[int(orig_idx)] = ll_value
        else:
            ll_ref_by_orig[int(orig_idx)] = ll_value

    for orig_idx in ll_ref_by_orig:
        if orig_idx in ll_alt_by_orig:
            results[orig_idx] = ll_alt_by_orig[orig_idx] - ll_ref_by_orig[orig_idx]

    return results


# -----------------------------
# PrefixLM single-position scoring
#   - now optionally returns ref/alt log-probs alongside log-odds
# -----------------------------
@torch.no_grad()
def compute_log_odds_prefixlm_single_pos(
    model,
    tokenizer,
    seqs,
    poses,
    refs,
    alts,
    max_len,
    return_components: bool = False,
):
    """
    If return_components is False (default, original behavior):
        returns List[Optional[float]]   (log_odds = log p(alt) - log p(ref) at position)

    If return_components is True:
        returns List[Optional[Dict[str, float]]] with keys:
            ref_log_prob, alt_log_prob, log_odds
    """
    n_in = len(seqs)
    if return_components:
        results: List[Optional[Dict[str, float]]] = [None] * n_in
    else:
        results: List[Optional[float]] = [None] * n_in

    device = next(model.parameters()).device

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

    # --- Pack: [CLS] wildtype [SEP] wildtype[:pos] [SEP] ---
    all_input_ids = []
    all_prefix_lengths = []
    all_logit_positions = []
    valid_batch_mask = []

    for seq, pos in zip(valid_seqs, valid_poses):
        enc_full = tokenizer(seq, add_special_tokens=False)["input_ids"]
        enc_prefix = enc_full
        enc_suffix = enc_full[:pos]

        packed = [cls_id] + enc_prefix + [sep_id] + enc_suffix + [sep_id]
        prefix_len = 1 + len(enc_prefix) + 1

        if len(packed) > max_len:
            valid_batch_mask.append(False)
            all_input_ids.append(None)
            all_prefix_lengths.append(None)
            all_logit_positions.append(None)
            continue

        logit_pos = len(packed) - 2

        all_input_ids.append(packed)
        all_prefix_lengths.append(prefix_len)
        all_logit_positions.append(logit_pos)
        valid_batch_mask.append(True)

    surviving = [
        (ids, plen, lpos, ref_id, alt_id, orig_idx)
        for ids, plen, lpos, ref_id, alt_id, orig_idx, keep in zip(
            all_input_ids, all_prefix_lengths, all_logit_positions,
            ref_ids, alt_ids, indices, valid_batch_mask
        )
        if keep
    ]

    if not surviving:
        return results

    ids_list, plens, lpos_list, rids, aids, orig_idxs = zip(*surviving)

    max_batch_len = max(len(ids) for ids in ids_list)
    padded_input_ids = []
    for ids in ids_list:
        padded_input_ids.append(ids + [pad_id] * (max_batch_len - len(ids)))

    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
    prefix_lengths_tensor = torch.tensor(plens, dtype=torch.long, device=device)

    base_model = model.module if hasattr(model, "module") else model

    if device.type == "cuda":
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = torch.autocast(device_type="cpu", enabled=False)

    with autocast_context:
        hidden_states = run_encoder_flash(
            model, input_ids_tensor, prefix_lengths_tensor, device
        )
        logits = base_model.decoder(base_model.head(hidden_states))
        logits_shifted = logits[:, :-1, :].contiguous()

    for batch_idx, (lpos, ref_id, alt_id, orig_idx) in enumerate(
        zip(lpos_list, rids, aids, orig_idxs)
    ):
        if lpos >= logits_shifted.shape[1]:
            continue

        logit_at_pos = logits_shifted[batch_idx, lpos, :]
        log_probs = torch.nn.functional.log_softmax(logit_at_pos.float(), dim=-1)

        ref_lp = float(log_probs[ref_id].item())
        alt_lp = float(log_probs[alt_id].item())
        log_odds = alt_lp - ref_lp

        if return_components:
            results[orig_idx] = {
                "ref_log_prob": ref_lp,
                "alt_log_prob": alt_lp,
                "log_odds": log_odds,
            }
        else:
            results[orig_idx] = log_odds

    return results


# -----------------------------
# Evaluation
# -----------------------------
def run_vep_eval(
    df: pd.DataFrame,
    model,
    tokenizer,
    batch_size: int,
    max_len: int,
    step_id: int,
    score_type: str,
    collect_per_variant: bool = False,
) -> Tuple[Dict[str, object], Optional[List[Dict[str, Any]]]]:
    """
    Returns (summary_dict, per_variant_rows_or_None).

    per_variant_rows is only populated when collect_per_variant=True
    AND score_type == 'single_pos'. Otherwise it is None.
    """
    if score_type not in {"full_ll", "single_pos"}:
        raise ValueError("score_type must be one of: full_ll, single_pos")

    print(f"Starting PrefixLM {score_type} VEP eval @ step {step_id}", flush=True)

    seqs = df["sequence"].tolist()
    poses = df["pos"].to_numpy(dtype=np.int64)
    refs = df["ref"].tolist()
    alts = df["alt"].tolist()
    labels = df["label"].to_numpy(dtype=np.int8)

    n = len(labels)
    preds = np.full(n, np.nan, dtype=np.float32)

    # Per-variant collection only makes sense for single_pos.
    do_components = bool(collect_per_variant) and (score_type == "single_pos")
    per_variant_rows: Optional[List[Dict[str, Any]]] = [] if do_components else None

    was_training = model.training
    model.eval()

    start_time = time.time()

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_end = min(i + batch_size, n)

            batch_seqs = seqs[i:batch_end]
            batch_poses = poses[i:batch_end]
            batch_refs = refs[i:batch_end]
            batch_alts = alts[i:batch_end]

            if score_type == "full_ll":
                scores = compute_log_odds_prefixlm_full_LL(
                    model=model,
                    tokenizer=tokenizer,
                    seqs=batch_seqs,
                    poses=batch_poses,
                    refs=batch_refs,
                    alts=batch_alts,
                    max_len=max_len,
                )
            else:
                scores = compute_log_odds_prefixlm_single_pos(
                    model=model,
                    tokenizer=tokenizer,
                    seqs=batch_seqs,
                    poses=batch_poses,
                    refs=batch_refs,
                    alts=batch_alts,
                    max_len=max_len,
                    return_components=do_components,
                )

            for j, score in enumerate(scores):
                variant_idx = i + j

                if score is None:
                    if do_components:
                        per_variant_rows.append({
                            "step": int(step_id),
                            "variant_idx": int(variant_idx),
                            "pos": int(batch_poses[j]),
                            "ref": batch_refs[j],
                            "alt": batch_alts[j],
                            "label": int(labels[variant_idx]),
                            "scored": 0,
                            "ref_log_prob": "",
                            "alt_log_prob": "",
                            "log_odds": "",
                            "model_score": "",
                        })
                    continue

                if do_components:
                    # score is a dict
                    log_odds = float(score["log_odds"])
                    model_score = -log_odds  # higher = more pathogenic
                    preds[variant_idx] = model_score
                    per_variant_rows.append({
                        "step": int(step_id),
                        "variant_idx": int(variant_idx),
                        "pos": int(batch_poses[j]),
                        "ref": batch_refs[j],
                        "alt": batch_alts[j],
                        "label": int(labels[variant_idx]),
                        "scored": 1,
                        "ref_log_prob": float(score["ref_log_prob"]),
                        "alt_log_prob": float(score["alt_log_prob"]),
                        "log_odds": log_odds,
                        "model_score": model_score,
                    })
                else:
                    # score is a float (LL_alt - LL_ref or single-pos log-odds)
                    preds[variant_idx] = -float(score)

            if (i % (batch_size * 20)) == 0:
                print(f"Progress: {i}/{n}", flush=True)

    if was_training:
        model.train()

    mask = ~np.isnan(preds)
    n_scored = int(mask.sum())

    if n_scored >= 10 and labels[mask].min() != labels[mask].max():
        auc = float(roc_auc_score(labels[mask], preds[mask]))
        print(f"AUC PrefixLM {score_type} at step {step_id}: {auc:.4f}", flush=True)
    else:
        auc = float("nan")
        print("Skipping AUC due to insufficient scored data or only one label class.", flush=True)

    elapsed_seconds = float(time.time() - start_time)
    print(f"[TIMER] VEP eval took {elapsed_seconds:.2f} seconds", flush=True)

    summary = {
        "step_id": int(step_id),
        "n_total": int(n),
        "n_scored": int(n_scored),
        "score_type": score_type,
        f"auc_prefixlm_{score_type}": auc,
        "auc": auc,
        "elapsed_seconds": elapsed_seconds,
    }
    return summary, per_variant_rows


def main():
    parser = HfArgumentParser((EvalArguments,))
    (eval_args,) = parser.parse_args_into_dataclasses()

    if eval_args.score_type not in {"full_ll", "single_pos"}:
        raise ValueError("--score_type must be either 'full_ll' or 'single_pos'.")

    print(f"score_type: {eval_args.score_type}", flush=True)

    # Per-variant CSV is only supported for single_pos.
    write_per_variant = bool(eval_args.per_variant_csv)
    if write_per_variant and eval_args.score_type != "single_pos":
        print(
            "[WARN] --per_variant_csv was set but score_type is "
            f"'{eval_args.score_type}'. Per-variant CSV is only produced for "
            "score_type='single_pos'. Ignoring the flag.",
            flush=True,
        )
        write_per_variant = False

    if eval_args.device is not None:
        device = torch.device(eval_args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"model_ckpt: {eval_args.model_ckpt}", flush=True)
    print(f"zero_shot_csv: {eval_args.zero_shot_csv}", flush=True)
    print(f"max_len: {eval_args.max_len}", flush=True)
    print(f"batch_size: {eval_args.batch_size}", flush=True)
    print(f"run_name: {eval_args.run_name}", flush=True)
    print(f"out_dir: {eval_args.out_dir}", flush=True)
    print(f"per_variant_csv: {write_per_variant}", flush=True)
    print(f"device: {device}", flush=True)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(eval_args.model_ckpt)

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id.")
    if tokenizer.cls_token_id is None:
        raise ValueError("Tokenizer has no cls_token_id.")
    if tokenizer.sep_token_id is None:
        raise ValueError("Tokenizer has no sep_token_id.")

    model = load_prefixlm_model(eval_args.model_ckpt, tokenizer, device)

    df = pd.read_csv(eval_args.zero_shot_csv)

    required = {"sequence", "pos", "ref", "alt", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["pos"] = df["pos"].astype(int)

    results, per_variant_rows = run_vep_eval(
        df=df,
        model=model,
        tokenizer=tokenizer,
        batch_size=eval_args.batch_size,
        max_len=eval_args.max_len,
        step_id=eval_args.step_id,
        score_type=eval_args.score_type,
        collect_per_variant=write_per_variant,
    )

    model_dir_name, checkpoint_dir_name, csv_basename = _safe_name_from_ckpt_path(
        eval_args.model_ckpt,
        eval_args.score_type,
    )

    os.makedirs(eval_args.out_dir, exist_ok=True)
    out_path = os.path.join(eval_args.out_dir, csv_basename)

    row = {
        "model_ckpt": eval_args.model_ckpt,
        "model_dir": model_dir_name,
        "checkpoint_dir": checkpoint_dir_name,
        "zero_shot_csv": eval_args.zero_shot_csv,
        "run_name": eval_args.run_name,
        **results,
    }

    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"Wrote AUC results to: {out_path}", flush=True)

    # Per-variant CSV (single_pos only).
    if write_per_variant and per_variant_rows is not None:
        pv_basename = _per_variant_csv_basename(eval_args.model_ckpt)
        pv_path = os.path.join(eval_args.out_dir, pv_basename)
        pv_columns = [
            "step", "variant_idx", "pos", "ref", "alt", "label",
            "scored", "ref_log_prob", "alt_log_prob", "log_odds", "model_score",
        ]
        pd.DataFrame(per_variant_rows, columns=pv_columns).to_csv(pv_path, index=False)
        print(f"Wrote per-variant CSV to: {pv_path}", flush=True)


if __name__ == "__main__":
    main()