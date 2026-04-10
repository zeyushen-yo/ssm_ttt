#!/bin/bash
#SBATCH --job-name=eval_A200M
#SBATCH --account=henderson
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/runs/eval_phaseA_200M_%j.out

module load anaconda3/2025.12
module load cudatoolkit/12.6
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
export CUDA_HOME=/usr/local/cuda-12.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

DATA_DIR=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/data_cache_v2
RUNS_DIR=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/runs

echo "=== Phase A 200M Comprehensive Evaluation ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

python evaluate.py \
    --checkpoints \
        $RUNS_DIR/phaseA_2_200M/checkpoint_final.pt \
        $RUNS_DIR/phaseA_1_200M/checkpoint_final.pt \
        $RUNS_DIR/stage1_32kp_vanilla/checkpoint_final.pt \
        $RUNS_DIR/stage1_32kp_SWA/checkpoint_final.pt \
        $RUNS_DIR/stage1_32kp_C2/checkpoint_final.pt \
    --names \
        "A2_delta_current_200M" \
        "A1_hebb_sqrt_200M" \
        "Vanilla_SSM_200M" \
        "SWA_200M" \
        "C2_hebb_mean_200M" \
    --data_dir $DATA_DIR \
    --output $RUNS_DIR/phaseA_200M_eval.png \
    --context_lengths 2048 4096 8192 16384 32768 \
    --max_docs 50 \
    --ttt_onoff \
    --random_prefix_control \
    --shuffle_prefix_control

echo ""
echo "=== Evaluation Complete ==="
echo "Date: $(date)"
