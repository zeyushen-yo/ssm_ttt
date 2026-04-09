#!/bin/bash
#SBATCH --job-name=s2_ttt
#SBATCH --account=henderson
#SBATCH --partition=ailab
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@120
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/runs/%j_%x.out

# Usage: sbatch --export=CONFIG=configs/s2_decay.yaml --job-name=s2_decay scripts/run_ailab.sh

module load anaconda3/2025.12
module load cudatoolkit/12.6
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
export CUDA_HOME=/usr/local/cuda-12.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
python train.py --config $CONFIG &
PID=$!

forward_signal() {
    echo "Signal received at $(date), forwarding USR1 to PID=$PID"
    kill -USR1 $PID 2>/dev/null
}
trap forward_signal USR1 TERM INT

while kill -0 $PID 2>/dev/null; do
    wait $PID 2>/dev/null
done

wait $PID 2>/dev/null
RC=$?
echo ""
echo "=== Training Complete (exit=$RC) ==="
