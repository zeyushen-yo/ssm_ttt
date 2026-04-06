#!/bin/bash
#SBATCH --job-name=tiny_pilot
#SBATCH --account=henderson
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/tiny_pilot_%j_%x.out

# Usage: sbatch --export=CONFIG=configs/tiny_pilot_ssm_ttt.yaml scripts/run_tiny_pilot.sh

module load anaconda3/2025.12
module load cudatoolkit/12.6
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
export CUDA_HOME=/usr/local/cuda-12.6

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

echo "=== Config: $CONFIG ==="
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== Starting Training ==="
python train.py --config $CONFIG
