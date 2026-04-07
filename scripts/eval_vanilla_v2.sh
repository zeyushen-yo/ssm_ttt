#!/bin/bash
#SBATCH --job-name=eval_van_v2
#SBATCH --account=henderson
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/eval_van_v2_%j.out

module load anaconda3/2025.12
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
module load cudatoolkit/12.6
export CUDA_HOME=/usr/local/cuda-12.6

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

python evaluate.py \
    --checkpoints runs/stage1v2_vanilla/checkpoint_final.pt \
    --names vanilla_v2 \
    --data_dir data_cache_v2 \
    --output runs/stage1v2_vanilla_eval.png \
    --max_docs 50

echo "Evaluation complete: $(date)"
