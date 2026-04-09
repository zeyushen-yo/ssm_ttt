#!/bin/bash
#SBATCH --job-name=eval_phaseA
#SBATCH --account=henderson
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/runs/eval_phaseA_%j.out

module load anaconda3/2025.12
module load cudatoolkit/12.6
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
export CUDA_HOME=/usr/local/cuda-12.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

DATA_DIR=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/data_cache_v2
RUNS_DIR=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/runs

echo "=== Phase A Comprehensive Evaluation ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

python evaluate.py \
    --checkpoints \
        $RUNS_DIR/phaseA_0/checkpoint_final.pt \
        $RUNS_DIR/phaseA_1/checkpoint_final.pt \
        $RUNS_DIR/phaseA_2/checkpoint_final.pt \
        $RUNS_DIR/phaseA_3/checkpoint_final.pt \
        $RUNS_DIR/stage1_32kp_vanilla/checkpoint_final.pt \
        $RUNS_DIR/stage1_32kp_SWA/checkpoint_final.pt \
        $RUNS_DIR/stage1_32kp_C2/checkpoint_final.pt \
    --names \
        "A0_hebb_mean" \
        "A1_hebb_sqrt" \
        "A2_delta_current" \
        "A3_delta_base" \
        "Vanilla_SSM_200M" \
        "SWA_200M" \
        "C2_hebb_200M" \
    --data_dir $DATA_DIR \
    --output $RUNS_DIR/phaseA_eval.png \
    --context_lengths 2048 4096 8192 16384 32768 \
    --max_docs 50 \
    --ttt_onoff \
    --random_prefix_control \
    --shuffle_prefix_control

echo ""
echo "=== Evaluation Complete ==="
echo "Date: $(date)"
