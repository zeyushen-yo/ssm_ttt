#!/bin/bash
#SBATCH --job-name=s1_eval_C4
#SBATCH --account=henderson
#SBATCH --time=59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/s1_eval_%j.out

module load anaconda3/2025.12
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
module load cudatoolkit/12.6
export CUDA_HOME=/usr/local/cuda-12.6

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

python evaluate.py \
    --checkpoints /scratch/gpfs/HENDERSON/zs7353/ssm_ttt/runs/stage1_C4/checkpoint_final.pt \
    --names C4_layer_input \
    --data_dir /scratch/gpfs/HENDERSON/zs7353/ssm_ttt/data_cache_v2 \
    --output /scratch/gpfs/HENDERSON/zs7353/ssm_ttt/runs/stage1_eval_C4.png \
    --max_docs 50
