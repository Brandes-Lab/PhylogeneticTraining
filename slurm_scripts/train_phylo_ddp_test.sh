#!/bin/bash
#SBATCH --job-name=modernBERT_1B_phylo_aligned_ddp
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=20:00:00
#SBATCH --output=/gpfs/data/brandeslab/User/as12267/slurm_outputs/modernBERT_1B_phylo_aligned_ddp_%j.out
#SBATCH --error=/gpfs/data/brandeslab/User/as12267/slurm_outputs/modernBERT_1B_phylo_aligned_ddp_%j.err

# ──────────────────────────────────────────────────────────────────────
# Single-node DDP training launcher (1 node × 2 GPUs).
# model_type = ModernBERT, training_type = phylo_aligned -> PhyloTrainer
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# --- environment ---
module purge
module load cuda/12.6

source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=/gpfs/data/brandeslab/User/as12267/PhylogeneticTraining:${PYTHONPATH:-}

# Strip module-loaded CUDA from LD_LIBRARY_PATH to avoid cuDNN mismatches
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
  | grep -v '^/gpfs/share/apps/cuda/12.6' \
  | paste -sd: -)

echo "Python executable: $(which python)"
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "SLURM_NNODES:    $SLURM_NNODES"
echo "SLURM_NODELIST:  $SLURM_JOB_NODELIST"
echo "SLURM_JOB_ID:    $SLURM_JOB_ID"

# --- rendezvous (single node -> master is the local node) ---
if command -v scontrol &>/dev/null; then
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
else
  export MASTER_ADDR=$(hostname -s)   # fallback when scontrol isn't on PATH
fi
export MASTER_PORT=$((12000 + RANDOM % 20000))

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# --- NCCL / networking ---
export NCCL_SOCKET_IFNAME=ib,eth
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# --- PyTorch allocator ---
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- caches / logging ---
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TOKENIZERS_PARALLELISM=false

# --- WandB ---
export WANDB_API_KEY="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"

# --- launch (1 node × 2 GPUs) ---
torchrun \
  --nnodes=1 \
  --nproc-per-node=2 \
  --rdzv-id=$SLURM_JOB_ID \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  /gpfs/data/brandeslab/User/as12267/PhylogeneticTraining/python_scripts/train_modernBERT.py \
  --run_name modernBERT_1B_phylo_aligned_ddp_${SLURM_JOB_ID} \
  --model_type "ModernBERT" \
  --training_type "phylo_aligned" \
  --wandb_project "phylo-llm" \
  --tokenizer_path ./phylo_char_tokenizer_with_bos \
  --train_dataset_type "uniref90_arrow_lmdb" \
  --max_position_embeddings 2048 \
  --train_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/train \
  --lmdb_path /gpfs/data/brandeslab/Data/uniref/uniref100_merged.lmdb \
  --val_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/test \
  --vep_input_csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
  --output_dir /gpfs/data/brandeslab/phylo_llm_checkpts \
  --attn_implementation flash_attention_2 \
  --num_train_epochs 100 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 128 \
  --vep_batch_size 16 \
  --learning_rate 3e-4 \
  --logging_steps 4 \
  --vep_eval_steps 500 \
  --dataloader_num_workers 16 \
  --dataloader_persistent_workers True \
  --dataloader_prefetch_factor 8 \
  --eval_strategy "no" \
  --save_strategy "no" \
  --eval_strategy "steps" \
  --eval_steps 500 \
  --ddp_timeout 3600