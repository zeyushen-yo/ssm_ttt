#!/bin/bash
#SBATCH --job-name=s1_eval_v2
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

echo "Evaluating vanilla, C0, C1, C2, C5"
echo "Started: $(date)"

python evaluate.py \
    --checkpoints \
        ${RUNS_DIR}/stage1_vanilla/checkpoint_final.pt \
        ${RUNS_DIR}/stage1_C0/checkpoint_final.pt \
        ${RUNS_DIR}/stage1_C1/checkpoint_final.pt \
        ${RUNS_DIR}/stage1_C2/checkpoint_final.pt \
        ${RUNS_DIR}/stage1_C5/checkpoint_final.pt \
    --names vanilla C0_old_rule C1_norm_c128 C2_norm_c64 C5_late_layers \
    --data_dir "$DATA_DIR" \
    --output "${RUNS_DIR}/stage1_eval_5models.png" \
    --max_docs 50

echo "Evaluation complete: $(date)"
