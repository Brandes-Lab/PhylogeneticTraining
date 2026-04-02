#!/bin/bash
#SBATCH --job-name=t5gemma_phylo_custom
#SBATCH --partition=a100_dev
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=phylo-custom-training-%j.out
#SBATCH --error=phylo-custom-training-%j.err

# === Load and activate conda environment ===
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126

# === Confirm CUDA Access ===
which python
python - <<'EOF'
import torch
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
EOF

echo "Python executable: $(which python)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
nvidia-smi

# === Distributed setup ===
while true; do
  PORT=$(shuf -i 20000-25000 -n 1)
  netstat -tuln | grep -q ":$PORT " || break
done
export MASTER_PORT=$PORT
echo "Selected free MASTER_PORT: $PORT"

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR=$MASTER_ADDR"

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$master_addr" hostname --ip-address)
echo "Head node IP: $head_node_ip"

# === Environment ===
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export WANDB_PROJECT=phylo-llm
export WANDB_API_KEY=ae9049d442db2ba3fa77f7928c1dae68353b3762
export TOKENIZERS_PARALLELISM=false

cd /gpfs/data/brandeslab/Project/HuggingfaceTransformer/
export PATH=/gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126/bin:$PATH

# === Train ===
time torchrun --nnodes=1 --nproc-per-node=1 --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --rdzv_endpoint=${head_node_ip}:${MASTER_PORT} --rdzv_backend=c10d python_scripts/train_t5gemma_custom.py --run_name t5gemma_phylo_custom --tokenizer_path ./phylo_char_tokenizer_updated --train_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/train --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx --dataset_type fasta --max_tokens 16384 --megabatch_size 2048 --gradient_accumulation_steps 32 --learning_rate 1e-4 --warmup_steps 500 --max_steps 10000000 --max_seq_len 512 --logging_steps 8 --save_steps 50000 --output_dir /gpfs/data/brandeslab/model_checkpts --num_workers 8 --prefetch_factor 8 --wandb_project phylo-llm --wandb_entity sinha-anushka12-na
