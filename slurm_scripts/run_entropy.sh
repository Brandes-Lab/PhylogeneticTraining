#!/bin/bash
#SBATCH --job-name=clinvar_dist_prefixlm_aligned
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=20:00:00
#SBATCH --output=clinvar_dist_prefixlm_aligned_%j.out
#SBATCH --error=clinvar_dist_prefixlm_aligned_%j.err

set -euo pipefail

# --- environment ---
module purge
module load cuda/12.6

source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=/gpfs/data/brandeslab/User/as12267/HuggingfaceTransformer:${PYTHONPATH:-}

export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
  | grep -v '^/gpfs/share/apps/cuda/12.6' \
  | paste -sd: -)

export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TOKENIZERS_PARALLELISM=false

echo "Python executable: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# --- run ---
python python_scripts/entropy.py \
    --checkpoint_dir /gpfs/data/brandeslab/phylo_llm_checkpts/modernBERT_113M_prefixlm_aligned_bs512_ctxt_2048 \
    --tokenizer_path /gpfs/data/brandeslab/User/as12267/HuggingfaceTransformer/phylo_char_tokenizer_with_bos \
    --vep_csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
    --output_csv /gpfs/data/brandeslab/User/as12267/modernBERT_113M_prefixlm_aligned_bs512_ctxt_2048/clinvar_full_dist_entropy.csv \
    --start_step 500 \
    --end_step 6500 \
    --step_size 500 \
    --batch_size 8 \
    --max_len 2048 \
    --compute_seq_metrics