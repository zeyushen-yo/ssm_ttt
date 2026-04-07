#!/bin/bash
#SBATCH --job-name=eval_32kp_all
#SBATCH --account=henderson
#SBATCH --time=59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/eval_32kp_all_%j.out

module load anaconda3/2025.12
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
module load cudatoolkit/12.6
export CUDA_HOME=/usr/local/cuda-12.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

DATA_DIR=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/data_cache_v2
RUNS_DIR=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/runs

CKPTS=""
NAMES=""

for cfg in SWA vanilla C1 C2; do
    dir=${RUNS_DIR}/stage1_32kp_${cfg}
    CKPT=${dir}/checkpoint_final.pt
    if [ ! -f "$CKPT" ]; then
        latest=$(ls -t ${dir}/checkpoint_*.pt 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            CKPT=$latest
            echo "Using latest checkpoint for $cfg: $CKPT"
        else
            echo "WARNING: No checkpoint for $cfg, skipping"
            continue
        fi
    fi
    CKPTS="$CKPTS $CKPT"
    NAMES="$NAMES ${cfg}_32kp"
done

echo "Evaluating: $NAMES"

python evaluate.py \
    --checkpoints $CKPTS \
    --names $NAMES \
    --data_dir "$DATA_DIR" \
    --output "${RUNS_DIR}/stage1_32kp_all_figure2.png" \
    --max_docs 50

echo "Evaluation complete: $(date)"
