"""
Custom training loop for T5Gemma phylo encoder-decoder.

Replaces HuggingFace Trainer with a bare PyTorch loop for full
visibility into where time is spent at each stage.

Usage:
    torchrun --nnodes=1 --nproc-per-node=1 \
        python_scripts/train_t5gemma_custom.py \
        --run_name t5gemma_phylo_custom \
        --tokenizer_path ./phylo_char_tokenizer_updated \
        --train_dataset_path /gpfs/data/brandeslab/Data/uniref/hf_pairs_uniref90_final/train.jsonl \
        --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
        --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
        --max_tokens 16384 \
        --megabatch_size 2048 \
        --gradient_accumulation_steps 32 \
        --learning_rate 1e-4 \
        --max_steps 10000000 \
        --max_seq_len 512
"""

import os
import sys
import time
import argparse
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gLM.models import ProteinT5GemmaModel
from gLM.tokenizers import PhyloTokenizerLoader
from gLM.collator import PhyloCollator
from gLM.dataset import Uniref90ArrowDatasetForFASTA, Uniref90ArrowDatasetForLMDB


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Simple cosine schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")

    # Data
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="fasta", choices=["fasta", "lmdb"])
    parser.add_argument("--fasta_path", type=str, default="")
    parser.add_argument("--index_db_path", type=str, default="")
    parser.add_argument("--lmdb_path", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=8)

    # Training
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Only used when --max_tokens is 0 (disabled). Otherwise ignored.")
    parser.add_argument("--max_tokens", type=int, default=0,
                        help="Token budget per micro-batch (total padded positions). 0 = disabled (use fixed --batch_size).")
    parser.add_argument("--megabatch_size", type=int, default=0,
                        help="Pool size for megabatch reshuffling. 0 = no reshuffling (greedy fill from random stream). "
                             "When >0, fetches this many pairs, sorts by (enc_len, dec_len), then packs token-budget batches.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=10_000_000)
    parser.add_argument("--logging_steps", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=50_000)
    parser.add_argument("--output_dir", type=str, default="/gpfs/data/brandeslab/model_checkpts")

    # W&B
    parser.add_argument("--run_name", type=str, default="t5gemma_phylo_custom")
    parser.add_argument("--wandb_project", type=str, default="phylo-llm")
    parser.add_argument("--wandb_entity", type=str, default="sinha-anushka12-na")
    parser.add_argument("--disable_wandb", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Distributed setup ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    def log(msg):
        if is_main:
            print(msg, flush=True)

    # ---- W&B ----
    if is_main and not args.disable_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, entity=args.wandb_entity)
    else:
        wandb.init(mode="disabled")

    # ---- Tokenizer ----
    tokenizer = PhyloTokenizerLoader(args.tokenizer_path)
    log(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # ---- Model ----
    model = ProteinT5GemmaModel(
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer,
        attn_implementation=args.attn_implementation,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_seq_len,
    ).build()
    model.config.decoder_start_token_id = tokenizer.pad_token_id

    if not args.disable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    log(f"Gradient checkpointing: {'DISABLED' if args.disable_gradient_checkpointing else 'ENABLED'}")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {num_params:,}")
    log(f"LM head shape: {model.lm_head.out_proj.weight.shape}")

    # ---- Count encoder/decoder params for MFU ----
    raw_model = model
    if hasattr(raw_model, "model") and hasattr(raw_model.model, "encoder"):
        enc_params = sum(p.numel() for p in raw_model.model.encoder.parameters())
        dec_params = sum(p.numel() for p in raw_model.model.decoder.parameters())
    else:
        enc_params = num_params // 2
        dec_params = num_params // 2
    gpu_peak_flops = 312e12  # A100 bf16 peak
    log(f"Encoder params: {enc_params:,} | Decoder params: {dec_params:,}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # ---- Dataset + DataLoader ----
    if args.dataset_type == "fasta":
        train_ds = Uniref90ArrowDatasetForFASTA(
            dataset_path=args.train_dataset_path,
            training_type="phylo_encoder_decoder",
            fasta_path=args.fasta_path,
            idx_db_path=args.index_db_path,
        )
    else:
        train_ds = Uniref90ArrowDatasetForLMDB(
            dataset_path=args.train_dataset_path,
            training_type="phylo_encoder_decoder",
            lmdb_path=args.lmdb_path,
        )
    log(f"Dataset size: {len(train_ds)}")

    collator = PhyloCollator(
        tokenizer=tokenizer,
        training_type="phylo_encoder_decoder",
        max_seq_len=args.max_seq_len,
    )

    use_token_budget = args.max_tokens > 0

    if use_token_budget:
        # DataLoader yields batches of raw (s1, s2) pairs; we re-batch manually
        # Use a reasonable batch_size so workers fetch in parallel efficiently
        _dl_batch_size = min(256, args.megabatch_size) if args.megabatch_size > 0 else 256
        dataloader = DataLoader(
            train_ds,
            batch_size=_dl_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
            worker_init_fn=seed_worker,
            collate_fn=lambda batch: batch,  # return list of raw (s1, s2) tuples
            drop_last=False,
        )
    else:
        dataloader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
            worker_init_fn=seed_worker,
            collate_fn=collator,
            drop_last=True,
        )

    # ---- Optimizer + Scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 doesn't need scaling, placeholder

    use_megabatch = use_token_budget and args.megabatch_size > 0

    # ---- Training loop ----
    if use_megabatch:
        log(f"Starting training: max_tokens={args.max_tokens}, megabatch_size={args.megabatch_size}, "
            f"grad_accum={args.gradient_accumulation_steps}")
    elif use_token_budget:
        log(f"Starting training: max_tokens={args.max_tokens}, grad_accum={args.gradient_accumulation_steps}")
    else:
        log(f"Starting training: batch_size={args.batch_size}, grad_accum={args.gradient_accumulation_steps}, "
            f"effective_batch={args.batch_size * args.gradient_accumulation_steps}")

    global_step = 0
    accum_loss = 0.0
    accum_tokens = 0
    optimizer.zero_grad()

    # Timing accumulators (reset every logging_steps)
    t_data_total = 0.0
    t_forward_total = 0.0
    t_backward_total = 0.0
    t_optim_total = 0.0
    micro_steps_since_log = 0

    # Batch stats accumulators (reset every logging_steps)
    accum_samples = 0
    accum_enc_real = 0
    accum_enc_padded = 0
    accum_dec_real = 0
    accum_dec_padded = 0
    accum_flops = 0.0

    train_start = time.time()

    def _run_micro_step(batch):
        """Forward + backward on one micro-batch."""
        nonlocal accum_loss, accum_tokens, t_forward_total, t_backward_total, micro_steps_since_log
        nonlocal accum_samples, accum_enc_real, accum_enc_padded, accum_dec_real, accum_dec_padded, accum_flops

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # Track batch shape stats before forward
        bs = batch["input_ids"].shape[0]
        padded_enc_len = batch["input_ids"].shape[1]
        padded_dec_len = batch["labels"].shape[1]
        enc_real = batch["attention_mask"].sum().item()
        dec_real = (batch["labels"] != -100).sum().item()

        accum_samples += bs
        accum_enc_real += enc_real
        accum_enc_padded += bs * padded_enc_len
        accum_dec_real += dec_real
        accum_dec_padded += bs * padded_dec_len

        torch.cuda.synchronize()
        t_fwd_start = time.time()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
        torch.cuda.synchronize()
        t_fwd_end = time.time()

        loss.backward()
        torch.cuda.synchronize()
        t_bwd_end = time.time()

        step_time = t_bwd_end - t_fwd_start
        # FLOPs: 6 * params * tokens for fwd+bwd per component
        flops = 6 * (enc_params * enc_real + dec_params * dec_real)
        accum_flops += flops

        accum_loss += loss.item() * args.gradient_accumulation_steps
        accum_tokens += dec_real

        t_forward_total += (t_fwd_end - t_fwd_start)
        t_backward_total += (t_bwd_end - t_fwd_end)
        micro_steps_since_log += 1

    def _maybe_step_and_log():
        """Optimizer step + logging every gradient_accumulation_steps micro-steps."""
        nonlocal global_step, accum_loss, accum_tokens
        nonlocal t_data_total, t_forward_total, t_backward_total, t_optim_total, micro_steps_since_log
        nonlocal accum_samples, accum_enc_real, accum_enc_padded, accum_dec_real, accum_dec_padded, accum_flops

        if micro_steps_since_log % args.gradient_accumulation_steps != 0:
            return

        t_opt_start = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t_opt_end = time.time()
        t_optim_total += (t_opt_end - t_opt_start)

        global_step += 1

        if global_step % args.logging_steps == 0:
            n = micro_steps_since_log
            avg_loss = accum_loss / n if n > 0 else 0
            elapsed = time.time() - train_start
            lr = scheduler.get_last_lr()[0]
            total_time = t_data_total + t_forward_total + t_backward_total + t_optim_total

            # Batch stats
            avg_samples_per_micro = accum_samples / n if n > 0 else 0
            enc_pad_waste = 1 - (accum_enc_real / accum_enc_padded) if accum_enc_padded > 0 else 0
            dec_pad_waste = 1 - (accum_dec_real / accum_dec_padded) if accum_dec_padded > 0 else 0
            total_pad_waste = 1 - ((accum_enc_real + accum_dec_real) / (accum_enc_padded + accum_dec_padded)) if (accum_enc_padded + accum_dec_padded) > 0 else 0

            # MFU: actual FLOPS / peak FLOPS over the fwd+bwd time
            mfu = accum_flops / (total_time * gpu_peak_flops) if total_time > 0 else 0
            tflops_per_sec = accum_flops / total_time / 1e12 if total_time > 0 else 0

            log(f"step {global_step} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                f"elapsed {elapsed/3600:.2f}h")
            log(f"  BATCH: {avg_samples_per_micro:.0f} samples/micro | "
                f"enc_real {accum_enc_real/n:.0f} dec_real {accum_dec_real/n:.0f} tokens/micro")
            log(f"  PADDING WASTE: enc {enc_pad_waste:.1%} | dec {dec_pad_waste:.1%} | "
                f"total {total_pad_waste:.1%}")
            log(f"  MFU: {mfu:.2%} | {tflops_per_sec:.1f} TFLOPS/s")
            log(f"  TIMING ({n} micro-steps): "
                f"data {t_data_total:.2f}s ({t_data_total/n*1000:.0f}ms/step) | "
                f"fwd {t_forward_total:.2f}s ({t_forward_total/n*1000:.0f}ms/step) | "
                f"bwd {t_backward_total:.2f}s ({t_backward_total/n*1000:.0f}ms/step) | "
                f"optim {t_optim_total:.2f}s | "
                f"total {total_time:.2f}s")

            if is_main and not args.disable_wandb:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr": lr,
                    "train/global_step": global_step,
                    "batch/samples_per_micro": avg_samples_per_micro,
                    "batch/enc_real_tokens_per_micro": accum_enc_real / n,
                    "batch/dec_real_tokens_per_micro": accum_dec_real / n,
                    "batch/enc_padding_waste": enc_pad_waste,
                    "batch/dec_padding_waste": dec_pad_waste,
                    "batch/total_padding_waste": total_pad_waste,
                    "perf/mfu": mfu,
                    "perf/tflops_per_sec": tflops_per_sec,
                    "perf/data_ms_per_microstep": t_data_total / n * 1000,
                    "perf/fwd_ms_per_microstep": t_forward_total / n * 1000,
                    "perf/bwd_ms_per_microstep": t_backward_total / n * 1000,
                }, step=global_step)

            accum_loss = 0.0
            accum_tokens = 0
            accum_samples = 0
            accum_enc_real = 0
            accum_enc_padded = 0
            accum_dec_real = 0
            accum_dec_padded = 0
            accum_flops = 0.0
            t_data_total = 0.0
            t_forward_total = 0.0
            t_backward_total = 0.0
            t_optim_total = 0.0
            micro_steps_since_log = 0

        if global_step % args.save_steps == 0 and is_main:
            save_dir = os.path.join(args.output_dir, args.run_name, f"step_{global_step}")
            os.makedirs(save_dir, exist_ok=True)
            raw_model = model.module if hasattr(model, "module") else model
            raw_model.save_pretrained(save_dir)
            tokenizer.tokenizer.save_pretrained(save_dir)
            log(f"Saved checkpoint to {save_dir}")

    def _pack_batches_from_sorted_pool(pool):
        """Greedily pack token-budget batches from a length-sorted pool.

        Cost model: total padded positions = batch_size × (max_enc_len + max_dec_len).
        A new pair is added only if the resulting total stays within max_tokens.
        """
        batches = []
        current_batch = []
        max_enc = 0
        max_dec = 0

        for pair in pool:
            enc_len = min(len(pair[0]), args.max_seq_len)
            dec_len = min(len(pair[1]), args.max_seq_len)

            new_max_enc = max(max_enc, enc_len)
            new_max_dec = max(max_dec, dec_len)
            new_cost = (len(current_batch) + 1) * (new_max_enc + new_max_dec)

            if current_batch and new_cost > args.max_tokens:
                batches.append(current_batch)
                current_batch = [pair]
                max_enc = enc_len
                max_dec = dec_len
            else:
                current_batch.append(pair)
                max_enc = new_max_enc
                max_dec = new_max_dec

        if current_batch:
            batches.append(current_batch)

        return batches

    while global_step < args.max_steps:
        t_iter_start = time.time()

        if use_megabatch:
            # --- Megabatch reshuffling ---
            # 1. Collect a pool of megabatch_size random pairs
            pool = []
            for batch_of_pairs in dataloader:
                pool.extend(batch_of_pairs)
                if len(pool) >= args.megabatch_size:
                    break

            # 2. Sort by (enc_len, dec_len) — groups similar input AND output lengths
            pool.sort(key=lambda pair: (min(len(pair[0]), args.max_seq_len),
                                        min(len(pair[1]), args.max_seq_len)))

            # 3. Pack into token-budget batches
            micro_batches = _pack_batches_from_sorted_pool(pool)

            t_data_total += time.time() - t_iter_start

            # 4. Shuffle batch order to avoid always training short→long
            random.shuffle(micro_batches)

            # 5. Process each micro-batch
            for mb in micro_batches:
                batch = collator(mb)
                _run_micro_step(batch)
                _maybe_step_and_log()

                if global_step >= args.max_steps:
                    break

        elif use_token_budget:
            # --- Token-budget batching without reshuffling ---
            buffer = []
            budget_used = 0
            done = False

            for batch_of_pairs in dataloader:
                for sample in batch_of_pairs:
                    s1, s2 = sample
                    cost = min(max(len(s1), len(s2)), args.max_seq_len)

                    if buffer and budget_used + cost > args.max_tokens:
                        t_data_total += time.time() - t_iter_start

                        batch = collator(buffer)
                        _run_micro_step(batch)
                        _maybe_step_and_log()

                        if global_step >= args.max_steps:
                            done = True
                            break

                        buffer = []
                        budget_used = 0
                        t_iter_start = time.time()

                    buffer.append(sample)
                    budget_used += cost

                if done:
                    break

            if buffer and global_step < args.max_steps:
                t_data_total += time.time() - t_iter_start
                batch = collator(buffer)
                _run_micro_step(batch)
                _maybe_step_and_log()

        else:
            # --- Original fixed batch_size path ---
            for batch in dataloader:
                t_data_total += time.time() - t_iter_start

                _run_micro_step(batch)
                _maybe_step_and_log()

                if global_step >= args.max_steps:
                    break

                t_iter_start = time.time()

    log(f"Training complete. {global_step} steps in {(time.time() - train_start)/3600:.2f}h")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
