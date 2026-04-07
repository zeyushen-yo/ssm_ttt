#!/bin/bash
#SBATCH --job-name=eval_done_v2
#SBATCH --account=henderson
#SBATCH --time=59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/eval_done_v2_%j.out

module load anaconda3/2025.12
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
module load cudatoolkit/12.6
export CUDA_HOME=/usr/local/cuda-12.6

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

CKPTS=""
NAMES=""
for cfg in SWA vanilla C0 C1; do
    RUN_DIR=runs/stage1v2_${cfg}
    CKPT=${RUN_DIR}/checkpoint_final.pt
    if [ ! -f "$CKPT" ]; then
        CKPT=$(ls -t ${RUN_DIR}/checkpoint_*.pt 2>/dev/null | head -1)
    fi
    if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
        CKPTS="$CKPTS $CKPT"
        NAMES="$NAMES $cfg"
    else
        echo "WARNING: No checkpoint found for $cfg"
    fi
done

echo "Evaluating: $NAMES"
python evaluate.py \
    --checkpoints $CKPTS \
    --names $NAMES \
    --data_dir data_cache_v2 \
    --output runs/stage1v2_partial_figure2.png \
    --max_docs 50

echo "Evaluation complete: $(date)"
