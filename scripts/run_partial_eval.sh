#!/bin/bash
#SBATCH --job-name=s1_eval_partial
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

DATA_DIR=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/data_cache_v2
RUNS_DIR=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/runs

echo "Evaluating vanilla + C0 (partial eval while others train)"
echo "Started: $(date)"

python evaluate.py \
    --checkpoints \
        ${RUNS_DIR}/stage1_vanilla/checkpoint_final.pt \
        ${RUNS_DIR}/stage1_C0/checkpoint_final.pt \
    --names vanilla C0 \
    --data_dir "$DATA_DIR" \
    --output "${RUNS_DIR}/stage1_partial_figure2.png" \
    --max_docs 50

echo "Evaluation complete: $(date)"
