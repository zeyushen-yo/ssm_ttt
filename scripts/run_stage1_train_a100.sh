#!/bin/bash
#SBATCH --job-name=s1_train
#SBATCH --account=henderson
#SBATCH --time=59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --signal=B:USR1@120
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/s1_train_%j.out

module load anaconda3/2025.12
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
module load cudatoolkit/12.6
export CUDA_HOME=/usr/local/cuda-12.6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

CONFIG=$1
echo "Stage 1 Training (A100): $CONFIG"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name())' 2>/dev/null)"

python train.py --config "$CONFIG"

echo "Training complete: $(date)"
