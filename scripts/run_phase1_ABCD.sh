#!/bin/bash
#SBATCH --job-name=phase1_ABCD
#SBATCH --account=henderson
#SBATCH --time=59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:4
#SBATCH --signal=B:USR1@120
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/phase1_ABCD_%j.out

module load anaconda3/2025.12
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
module load cudatoolkit/12.6
export CUDA_HOME=/usr/local/cuda-12.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

echo "Node: $(hostname)"
echo "Started: $(date)"

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/phase1_A.yaml &
PID_A=$!

CUDA_VISIBLE_DEVICES=1 python train.py --config configs/phase1_B.yaml &
PID_B=$!

CUDA_VISIBLE_DEVICES=2 python train.py --config configs/phase1_C.yaml &
PID_C=$!

CUDA_VISIBLE_DEVICES=3 python train.py --config configs/phase1_D.yaml &
PID_D=$!

echo "A=$PID_A B=$PID_B C=$PID_C D=$PID_D"

forward_signal() {
    echo "Signal received at $(date)"
    kill -USR1 $PID_A 2>/dev/null
    kill -USR1 $PID_B 2>/dev/null
    kill -USR1 $PID_C 2>/dev/null
    kill -USR1 $PID_D 2>/dev/null
}
trap forward_signal USR1 TERM INT

while kill -0 $PID_A 2>/dev/null || kill -0 $PID_B 2>/dev/null || kill -0 $PID_C 2>/dev/null || kill -0 $PID_D 2>/dev/null; do
    wait $PID_A $PID_B $PID_C $PID_D 2>/dev/null
done

wait $PID_A 2>/dev/null; RC_A=$?
wait $PID_B 2>/dev/null; RC_B=$?
wait $PID_C 2>/dev/null; RC_C=$?
wait $PID_D 2>/dev/null; RC_D=$?

echo "A exit=$RC_A, B exit=$RC_B, C exit=$RC_C, D exit=$RC_D"
echo "All done: $(date)"
