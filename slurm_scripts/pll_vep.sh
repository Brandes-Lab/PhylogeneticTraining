export SLURM_CPUS_PER_TASK=8
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

for i in $(seq 0 7); do
  export SLURM_ARRAY_TASK_ID=$i
  export MASTER_PORT=$((12000 + RANDOM % 20000))
  CKPT_PATH=$(sed -n "$((i+1))p" checkpoints.txt)
  echo "=== Running checkpoint $i: ${CKPT_PATH} ==="
  # torchrun --nproc_per_node=1 --master_port=$MASTER_PORT python_scripts/pll.py \
  #   --model_ckpt "${CKPT_PATH}" \
  #   --zero_shot_csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
  #   --max_len 1024 \
  #   --batch_size 128 \
  #   --pll_mode singlepos \
  #   --run_name "pll_$(basename "$(dirname "${CKPT_PATH}")")_$(basename "${CKPT_PATH}")" \
  #   --out_dir /gpfs/data/brandeslab/User/as12267/T5Gemma_97M_phylo_bs_4096_arrow_fasta_file_zero_shot_vep_singlepos_wtenc
  torchrun --nproc_per_node=1 --master_port=$MASTER_PORT python_scripts/pll.py \
  --model_ckpt "${CKPT_PATH}" \
  --zero_shot_csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
  --max_len 1024 \
  --batch_size 128 \
  --pll_mode wtenc \
  --run_name "pll_$(basename "$(dirname "${CKPT_PATH}")")_$(basename "${CKPT_PATH}")" \
  --out_dir /gpfs/data/brandeslab/User/as12267/T5Gemma_97M_phylo_bs_4096_arrow_fasta_file_zero_shot_vep_full_seq_LL_wtenc

done