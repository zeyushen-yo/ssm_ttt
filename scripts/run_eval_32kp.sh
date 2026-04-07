#!/bin/bash
#SBATCH --job-name=eval_32kp
#SBATCH --account=henderson
#SBATCH --time=59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/eval_32kp_%j.out

module load anaconda3/2025.12
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
module load cudatoolkit/12.6
export CUDA_HOME=/usr/local/cuda-12.6

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

DATA_DIR=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/data_cache_v2
RUNS_DIR=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/runs

CKPTS=""
NAMES=""

for cfg in SWA vanilla; do
    CKPT=${RUNS_DIR}/stage1_32kp_${cfg}/checkpoint_final.pt
    if [ -f "$CKPT" ]; then
        CKPTS="$CKPTS $CKPT"
        NAMES="$NAMES ${cfg}_32kp"
    else
        echo "WARNING: No checkpoint_final.pt for $cfg"
    fi
done

echo "Evaluating: $NAMES"

python evaluate.py \
    --checkpoints $CKPTS \
    --names $NAMES \
    --data_dir "$DATA_DIR" \
    --output "${RUNS_DIR}/stage1_32kp_figure2.png" \
    --max_docs 50

echo "Evaluation complete: $(date)"
