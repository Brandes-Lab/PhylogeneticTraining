#!/bin/bash
#SBATCH --job-name=modernBERT_phylo_aligned_ddp
#SBATCH --partition=a100_long
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=07-00:00:00
#SBATCH --output=modernBERT_phylo_aligned_ddp_%j.out
#SBATCH --error=modernBERT_phylo_aligned_ddp_%j.err

# ──────────────────────────────────────────────────────────────────────
# Multi-node / multi-GPU DDP training launcher.
# model_type = ModernBERT, training_type = phylo_aligned -> PhyloTrainer
#
# For single-node testing (1 node × N GPUs), edit:
#   #SBATCH --nodes=1
#   #SBATCH --gres=gpu:N
# and adjust --nproc-per-node=N in the srun command.
# $SLURM_NNODES is read at runtime so nnodes self-adjusts.
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# --- environment ---
module purge
module load cuda/12.6

source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=/gpfs/data/brandeslab/User/as12267/HuggingfaceTransformer:${PYTHONPATH:-}

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

# --- rendezvous (master = first node in the SLURM allocation) ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
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

# --- launch ---
# srun launches one task per node (SBATCH --ntasks-per-node=1); each task runs
# torchrun, which spawns --nproc-per-node worker processes on that node.
# Total processes = SLURM_NNODES × nproc_per_node
#
# Effective batch size = world_size × per_device_train_batch_size × grad_accum
#                      = (SLURM_NNODES × 4) × 16 × 2
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc-per-node=4 \
  --rdzv-id=$SLURM_JOB_ID \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  /gpfs/data/brandeslab/User/as12267/HuggingfaceTransformer/python_scripts/train_modernBERT.py \
  --run_name modernBERT_phylo_aligned_ddp_${SLURM_JOB_ID} \
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
  --seed 42 \
  --data_seed 42 \
  --max_steps 1000000 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-4 \
  --logging_steps 10 \
  --vep_eval_steps 5000 \
  --dataloader_num_workers 8 \
  --dataloader_persistent_workers True \
  --dataloader_prefetch_factor 4 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --ddp_timeout 3600
