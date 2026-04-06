#!/bin/bash
#SBATCH --job-name=prep_val_v2
#SBATCH --account=henderson
#SBATCH --time=59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/prep_val_v2_%j.out

module load anaconda3/2025.12
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

python data/prepare_data.py \
    --output_dir /scratch/gpfs/HENDERSON/zs7353/ssm_ttt/data_cache_v2 \
    --num_tokens 200000000 \
    --max_val_docs 200 \
    --min_val_doc_len 32768
