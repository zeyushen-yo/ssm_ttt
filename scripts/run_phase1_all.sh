#!/bin/bash
#SBATCH --job-name=phase1
#SBATCH --account=henderson
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/runs/phase1_%j_%x.out

# Usage: sbatch --export=CONFIG=configs/tiny_pilot_ssm_ttt.yaml --job-name=p1_ttt scripts/run_phase1_all.sh

module load anaconda3/2025.12
module load cudatoolkit/12.6
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
export CUDA_HOME=/usr/local/cuda-12.6

export PYTHONUNBUFFERED=1

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

echo "=== Config: $CONFIG ==="
cat $CONFIG
echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== Checking data ==="
DATA_DIR=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('data_dir',''))")
if [ -f "$DATA_DIR/train.bin" ]; then
    echo "Pre-tokenized data found at $DATA_DIR"
    ls -lh $DATA_DIR/
else
    echo "ERROR: Pre-tokenized data not found at $DATA_DIR"
    echo "Please run data/prepare_data.py first on the login node."
    exit 1
fi

echo ""
echo "=== Starting Training ==="
python train.py --config $CONFIG

echo ""
echo "=== Training Complete ==="
