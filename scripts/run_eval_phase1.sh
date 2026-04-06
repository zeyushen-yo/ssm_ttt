#!/bin/bash
#SBATCH --job-name=p1_eval
#SBATCH --account=henderson
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/runs/eval_phase1_%j.out

module load anaconda3/2025.12
module load cudatoolkit/12.6
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
export CUDA_HOME=/usr/local/cuda-12.6
export PYTHONUNBUFFERED=1

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Use TTT_CKPT env var if set, otherwise use checkpoint_final.pt
TTT_CKPT=${TTT_CKPT:-runs/tiny_pilot_ssm_ttt_v2/checkpoint_final.pt}
TTT_NAME=${TTT_NAME:-SSM+TTT-v2}
OUTPUT=${OUTPUT:-runs/figure2_phase1_v2.png}

echo "TTT checkpoint: $TTT_CKPT"

python evaluate.py \
    --checkpoints \
        runs/tiny_pilot_vanilla_ssm/checkpoint_final.pt \
        runs/tiny_pilot_transformer_swa/checkpoint_final.pt \
        $TTT_CKPT \
    --names "Vanilla-SSM" "Transformer-SWA" "$TTT_NAME" \
    --data_dir /scratch/gpfs/HENDERSON/zs7353/ssm_ttt/data_cache \
    --output $OUTPUT \
    --context_lengths 2048 4096 8192 16384 32768 \
    --scored_suffix_len 2048 \
    --max_docs 30

echo ""
echo "=== Evaluation Complete ==="
