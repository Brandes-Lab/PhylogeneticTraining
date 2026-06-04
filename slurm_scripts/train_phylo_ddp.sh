#!/bin/bash
#SBATCH --job-name=modernBERT_1B_phylo_aligned_ddp
#SBATCH --partition=reservation
#SBATCH --reservation=brandeslab_reservation
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=14-00:00:00
#SBATCH --output=/gpfs/data/brandeslab/User/as12267/slurm_outputs/modernBERT_1B_phylo_aligned_ddp_%j.out
#SBATCH --error=/gpfs/data/brandeslab/User/as12267/slurm_outputs/modernBERT_1B_phylo_aligned_ddp_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────

module purge
module load cuda/12.6

source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126

# Restore Slurm binaries to PATH (module purge removes srun on compute nodes)
export PATH="/cm/shared/apps/slurm/current/bin:$PATH"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/gpfs/data/brandeslab/User/as12267/PhylogeneticTraining:${PYTHONPATH:-}

# Remove module CUDA libraries to avoid cuDNN / CUDA mismatch
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
  | grep -v '^/gpfs/share/apps/cuda/12.6' \
  | paste -sd: -)

# ─────────────────────────────────────────────────────────────
# Debug info
# ─────────────────────────────────────────────────────────────

echo "Python executable: $(which python)"
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "SLURM_NNODES:    ${SLURM_NNODES}"
echo "SLURM_NODELIST:  ${SLURM_JOB_NODELIST}"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# ─────────────────────────────────────────────────────────────
# Rendezvous
# Works without scontrol.
# For your allocation: a100nv-[4001-4007] -> a100nv-4001
# ─────────────────────────────────────────────────────────────

if [[ "${SLURM_JOB_NODELIST}" =~ ^([a-zA-Z0-9._-]+)\[([0-9]+)-[0-9]+\]$ ]]; then
    MASTER_ADDR="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
elif [[ "${SLURM_JOB_NODELIST}" =~ ^([a-zA-Z0-9._-]+)\[([0-9]+),.*\]$ ]]; then
    MASTER_ADDR="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
else
    MASTER_ADDR="${SLURM_JOB_NODELIST}"
fi

export MASTER_ADDR
export MASTER_PORT=$((12000 + RANDOM % 20000))

echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"

# ─────────────────────────────────────────────────────────────
# NCCL / networking
# ─────────────────────────────────────────────────────────────

export NCCL_SOCKET_IFNAME=ib,eth
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ─────────────────────────────────────────────────────────────
# Runtime settings
# ─────────────────────────────────────────────────────────────

export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

export WANDB_API_KEY="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"

echo "Launching DDP training..."
echo "World size: $((SLURM_NNODES * 4))"

# ─────────────────────────────────────────────────────────────
# Launch
# One SLURM task per node.
# Each task runs torchrun.
# Each torchrun launches 4 GPU workers.
# Total workers = 7 nodes × 4 GPUs = 28 processes.
# ─────────────────────────────────────────────────────────────

srun \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task=64 \
  torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc-per-node=4 \
    --rdzv-id="${SLURM_JOB_ID}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    /gpfs/data/brandeslab/User/as12267/PhylogeneticTraining/python_scripts/train_modernBERT.py \
    --run_name modernBERT_1B_phylo_aligned_ddp_${SLURM_JOB_ID} \
    --model_type "ModernBERT" \
    --training_type "phylo_aligned" \
    --wandb_project "phylo-llm" \
    --tokenizer_path ./phylo_char_tokenizer_with_bos \
    --train_dataset_type "uniref90_arrow_lmdb" \
    --max_position_embeddings 8192 \
    --train_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/train \
    --lmdb_path /gpfs/data/brandeslab/Data/uniref/uniref100_merged.lmdb \
    --val_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/test \
    --vep_input_csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
    --output_dir /gpfs/data/brandeslab/phylo_llm_checkpts \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1000000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 5 \
    --vep_batch_size 8 \
    --vep_max_len 8192 \
    --learning_rate 1e-4 \
    --logging_steps 9 \
    --vep_eval_steps 500000 \
    --dataloader_num_workers 32 \
    --dataloader_persistent_workers True \
    --dataloader_prefetch_factor 16 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --ddp_timeout 3600