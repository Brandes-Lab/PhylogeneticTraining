# type: ignore
import os
from dataclasses import dataclass, field  # type: ignore
from typing import Literal, Optional
import torch  # type: ignore
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
import wandb
from gLM.callbacks import ZeroShotVEPEvaluationCallback
from gLM.models import ProteinBertModel
from gLM.models.protein_modernbert_phylo import ProteinModernBertPrefixLM, run_sanity_checks
from gLM.attention_mask import prefixlm_forward_flash
from gLM.tokenizers import PhyloTokenizerLoader
from gLM.collator import create_mlm_collator, PhyloCollator
from gLM.collator.prefixlm_collator import PrefixLMCollator
from gLM.dataset import (
    Uniref90ArrowDatasetForFASTA,
    Uniref90ArrowEvalDatasetForFASTA,
    Uniref90ArrowDatasetForLMDB,
    Uniref90ArrowEvalDatasetForLMDB,
)
from gLM.train_utils import PhyloTrainer
from gLM.train_utils.prefixlm_trainer import PrefixLMTrainer

if torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    print(f"✅ CUDA available, using device: {DEVICE}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ MPS available, using MPS")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ CUDA/MPS not available. Using CPU")


def print_rank0(*args, **kwargs):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs)


_VALID_TRAINING_TYPES = {
    "ModernBERT": {"MLM", "phylo_aligned"},
    "PrefixLM_ModernBERT": {"phylo_unaligned"},
}


@dataclass
class ModelArguments:
    model_type: Literal["ModernBERT", "PrefixLM_ModernBERT"] = field(
        default="PrefixLM_ModernBERT", metadata={"help": "Type of model to use"}
    )
    tokenizer_path: str = field(
        default="char_tokenizer", metadata={"help": "Path to the tokenizer directory"}
    )
    max_position_embeddings: int = field(
        default=8192, metadata={"help": "Maximum sequence length for the model"}
    )
    attn_implementation: Literal["flash_attention_2", "sdpa"] = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation to use (ModernBERT only)"},
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    run_name: str = field(
        default="modernBERT_1B",
        metadata={"help": "Name for the experiment run"},
    )
    output_dir: str = field(
        default="/gpfs/data/brandeslab/model_checkpts",
        metadata={"help": "Directory to save model checkpoints"},
    )
    max_steps: int = field(
        default=-1, metadata={"help": "Maximum number of training steps"}
    )
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Training batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=32, metadata={"help": "Number of gradient accumulation steps"}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Evaluation batch size per device"}
    )
    learning_rate: float = field(
        default=1e-3, metadata={"help": "Learning rate for training"}
    )
    logging_steps: int = field(
        default=32, metadata={"help": "Number of steps between logging"}
    )
    vep_eval_steps: int = field(
        default=10000, metadata={"help": "Number of steps between VEP evaluations"}
    )
    dataloader_num_workers: int = field(
        default=6, metadata={"help": "Number of dataloader workers"}
    )
    dataloader_persistent_workers: bool = field(
        default=True,
        metadata={"help": "Whether to use persistent dataloader workers"},
    )
    dataloader_prefetch_factor: Optional[int] = field(
        default=None,
        metadata={"help": "Number of batches to prefetch per worker"},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Masking probability for MLM training"}
    )
    training_type: Literal["MLM", "phylo_aligned", "phylo_unaligned"] = field(
        default="phylo_unaligned", metadata={"help": "Type of training to perform"}
    )
    ## DDP arguments
    ddp_backend: str = field(default="nccl", metadata={"help": "DDP backend"})
    ddp_timeout: int = field(default=1800, metadata={"help": "DDP timeout in seconds"})
    ddp_find_unused_parameters: Optional[bool] = field(default=False, metadata={"help": "Find unused parameters in DDP"})

    # Arguments that shouldn't be changed
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    eval_strategy: str = field(default="no")
    eval_steps: int = field(default=50000)
    logging_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=1_000_000)
    report_to: str = field(default="wandb")
    remove_unused_columns: bool = field(default=False)
    group_by_length: bool = field(default=False)
    length_column_name: str = field(default="length")
    include_num_input_tokens_seen: str = field(default="non_padding")
    lr_scheduler_type: str = field(default="linear")
    warmup_steps: int = field(default=0)


@dataclass
class DataArguments:
    train_dataset_type: Literal["uniref90_arrow_fasta", "uniref90_arrow_lmdb"] = field(
        default="uniref90_arrow_lmdb", metadata={"help": "Type of training dataset"}
    )
    train_dataset_path: str = field(
        default="/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train",
        metadata={"help": "Path to the training dataset"},
    )
    val_dataset_path: str = field(
        default="/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only/validation",
        metadata={"help": "Path to the validation dataset"},
    )
    vep_input_csv: str = field(
        default="/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv",
        metadata={"help": "Path to the VEP evaluation CSV file"},
    )
    fasta_path: str = field(
        default="/gpfs/data/brandeslab/Data/uniref/uniref100.fasta",
        metadata={"help": "Path to the FASTA file"},
    )
    index_db_path: str = field(
        default="/gpfs/data/brandeslab/User/as12267/uniref100.idx",
        metadata={"help": "Path to the SQLite index DB for FASTA"},
    )
    lmdb_path: str = field(
        default="/gpfs/data/brandeslab/Data/uniref/uniref100_bk.lmdb",
        metadata={"help": "Path to the LMDB file"},
    )


@dataclass
class WandbArguments:
    wandb_project: str = field(
        default="long_runs",
        metadata={"help": "Weights & Biases project name"},
    )
    wandb_entity: str = field(
        default="sinha-anushka12-na", metadata={"help": "Weights & Biases entity name"}
    )
    disable_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to disable WandB logging"},
    )


def main():
    print("RANK", os.environ.get("RANK"),
          "LOCAL_RANK", os.environ.get("LOCAL_RANK"),
          "WORLD_SIZE", os.environ.get("WORLD_SIZE"))

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments, WandbArguments)
    )
    model_args, data_args, training_args, wandb_args = parser.parse_args_into_dataclasses()

    if training_args.dataloader_num_workers == 0:
        training_args.dataloader_prefetch_factor = None
        training_args.dataloader_persistent_workers = False

    print(f"[Rank {training_args.local_rank}] MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"[Rank {training_args.local_rank}] MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    print(f"[Rank {training_args.local_rank}] NCCL_TIMEOUT: {os.environ.get('NCCL_TIMEOUT')}")

    if training_args.training_type not in _VALID_TRAINING_TYPES[model_args.model_type]:
        raise ValueError(
            f"training_type '{training_args.training_type}' is not valid for "
            f"model_type '{model_args.model_type}'. "
            f"Valid options: {sorted(_VALID_TRAINING_TYPES[model_args.model_type])}"
        )

    rank = int(os.environ.get("RANK", 0))
    if not wandb_args.disable_wandb and rank == 0:
        wandb.init(
            project=wandb_args.wandb_project,
            name=training_args.run_name,
            entity=wandb_args.wandb_entity,
        )
    else:
        wandb.init(mode="disabled")

    # Load tokenizer
    tokenizer = PhyloTokenizerLoader(model_args.tokenizer_path)
    print_rank0(f"Using tokenizer from: {model_args.tokenizer_path}")
    print_rank0("GAP ID:", tokenizer.convert_tokens_to_ids("[GAP]"))
    print_rank0("Mask ID:", tokenizer.mask_token_id)
    print_rank0("Non-GAP ID:", tokenizer.convert_tokens_to_ids("-"))
    print_rank0("Tokenizer vocab size:", tokenizer.vocab_size)

    # =========================================================================
    # Build model
    # =========================================================================
    if model_args.model_type == "ModernBERT":
        print_rank0("Building ModernBERT model...")
        model = ProteinBertModel(
            vocab_size=tokenizer.vocab_size,
            tokenizer=tokenizer,
            attn_implementation=model_args.attn_implementation,
        ).build()
    else:  # PrefixLM_ModernBERT
        print_rank0("Building PrefixLM_ModernBERT model...")
        model = ProteinModernBertPrefixLM(
            vocab_size=tokenizer.vocab_size,
            tokenizer=tokenizer,
            max_position_embeddings=model_args.max_position_embeddings,
        ).build()

    model.gradient_checkpointing_enable()
    model.to(DEVICE)
    print_rank0(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print_rank0("Hidden size:", model.config.hidden_size)
    print_rank0(f"max_position_embeddings: {model.config.max_position_embeddings}")

    training_args.output_dir = f"{training_args.output_dir}/{training_args.run_name}"

    # =========================================================================
    # Load dataset
    # =========================================================================
    if data_args.train_dataset_type == "uniref90_arrow_fasta":
        print_rank0("Using Uniref Arrow + FASTA dataset")
        train_ds = Uniref90ArrowDatasetForFASTA(
            dataset_path=data_args.train_dataset_path,
            training_type=training_args.training_type,
            fasta_path=data_args.fasta_path,
            idx_db_path=data_args.index_db_path,
        )
        val_ds = Uniref90ArrowEvalDatasetForFASTA(
            dataset_path=data_args.val_dataset_path,
            training_type=training_args.training_type,
            fasta_path=data_args.fasta_path,
            idx_db_path=data_args.index_db_path,
        )
    else:  # uniref90_arrow_lmdb
        print_rank0("Using Uniref90 Arrow + LMDB dataset")
        train_ds = Uniref90ArrowDatasetForLMDB(
            dataset_path=data_args.train_dataset_path,
            training_type=training_args.training_type,
            lmdb_path=data_args.lmdb_path,
        )
        val_ds = Uniref90ArrowEvalDatasetForLMDB(
            dataset_path=data_args.val_dataset_path,
            training_type=training_args.training_type,
            lmdb_path=data_args.lmdb_path,
        )
    print_rank0("Validation dataset size:", len(val_ds))

    # =========================================================================
    # Create collator
    # =========================================================================
    if training_args.training_type == "MLM":
        print_rank0(f"Using MLM collator (masking probability: {training_args.mlm_probability})")
        data_collator = create_mlm_collator(
            tokenizer,
            max_seq_len=model_args.max_position_embeddings,
            mlm_probability=training_args.mlm_probability,
        )
    elif training_args.training_type == "phylo_aligned":
        print_rank0("Using PhyloCollator (phylo_aligned)")
        data_collator = PhyloCollator(
            tokenizer=tokenizer,
            training_type="phylo_aligned",
            max_seq_len=model_args.max_position_embeddings,
        )
    else:  # phylo_unaligned
        print_rank0("Using PrefixLMCollator (phylo_unaligned) — format: [CLS] seq2 [SEP] seq1 [SEP]")
        data_collator = PrefixLMCollator(
            tokenizer=tokenizer,
            max_seq_len=model_args.max_position_embeddings,
        )

    # =========================================================================
    # Create trainer
    # =========================================================================
    if model_args.model_type == "PrefixLM_ModernBERT":
        print_rank0("Using PrefixLMTrainer")
        trainer = PrefixLMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        if rank == 0:
            print_rank0("\n--- PrefixLM sanity checks ---")
            batch_raw = [train_ds[i] for i in range(4)]
            batch_out = data_collator(batch_raw)
            model.eval()
            loss, logits = prefixlm_forward_flash(model, batch_out, DEVICE)
            print_rank0(f"Sanity forward pass — loss: {loss.item():.4f}")
            model.zero_grad()
            run_sanity_checks(model, batch_out, DEVICE)
            model.train()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    elif training_args.training_type == "phylo_aligned":
        print_rank0("Using PhyloTrainer")
        trainer = PhyloTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:  # MLM
        print_rank0("Using Trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    trainer.add_callback(
        ZeroShotVEPEvaluationCallback(
            tokenizer=tokenizer,
            input_csv=data_args.vep_input_csv,
            trainer=trainer,
            eval_every_n_steps=training_args.vep_eval_steps,
            training_type=training_args.training_type,
        )
    )

    trainer.train()


if __name__ == "__main__":
    main()
