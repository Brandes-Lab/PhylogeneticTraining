#!/bin/bash
#SBATCH --job-name=build_uniref_index
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=build_index_%j.out
#SBATCH --error=build_index_%j.err

source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126

python HuggingfaceTransformer/build_index.py