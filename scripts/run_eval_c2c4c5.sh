#!/bin/bash
#SBATCH --job-name=eval_c2c4c5
#SBATCH --account=henderson
#SBATCH --time=59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/eval_c2c4c5_%j.out

module load anaconda3/2025.12
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
module load cudatoolkit/12.6
export CUDA_HOME=/usr/local/cuda-12.6

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

DATA_DIR=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/data_cache_v2
RUNS_DIR=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/runs

python evaluate.py \
    --checkpoints \
        ${RUNS_DIR}/stage1v2_C2/checkpoint_final.pt \
        ${RUNS_DIR}/stage1v2_C4/checkpoint_final.pt \
        ${RUNS_DIR}/stage1v2_C5/checkpoint_final.pt \
    --names C2 C4 C5 \
    --data_dir "$DATA_DIR" \
    --output "${RUNS_DIR}/stage1v2_c2c4c5_figure2.png" \
    --max_docs 50

echo "Evaluation complete: $(date)"
