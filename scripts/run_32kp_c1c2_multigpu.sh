#!/bin/bash
#SBATCH --job-name=32kp_C1C2
#SBATCH --account=henderson
#SBATCH --time=59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH --signal=B:USR1@120
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/32kp_c1c2_%j.out

module load anaconda3/2025.12
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
module load cudatoolkit/12.6
export CUDA_HOME=/usr/local/cuda-12.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

echo "Node: $(hostname)"
echo "Started: $(date)"

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/stage1_32k_C1.yaml &
PID_C1=$!

CUDA_VISIBLE_DEVICES=1 python train.py --config configs/stage1_32k_C2.yaml &
PID_C2=$!

echo "C1=$PID_C1 C2=$PID_C2"

forward_signal() {
    echo "Signal received"
    kill -USR1 $PID_C1 2>/dev/null
    kill -USR1 $PID_C2 2>/dev/null
}
trap forward_signal USR1 TERM INT

while kill -0 $PID_C1 2>/dev/null || kill -0 $PID_C2 2>/dev/null; do
    wait $PID_C1 $PID_C2 2>/dev/null
done

wait $PID_C1 2>/dev/null; RC1=$?
wait $PID_C2 2>/dev/null; RC2=$?

echo "C1 exit=$RC1, C2 exit=$RC2"
echo "Both done: $(date)"
