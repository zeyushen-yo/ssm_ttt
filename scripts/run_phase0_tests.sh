#!/bin/bash
#SBATCH --job-name=phase0_tests
#SBATCH --account=henderson
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/phase0_tests_%j.out

module load anaconda3/2025.12
module load cudatoolkit/12.6
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
export CUDA_HOME=/usr/local/cuda-12.6

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

echo "=== Environment Check ==="
python -c "
import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
import mamba_ssm; print('Mamba SSM:', mamba_ssm.__version__)
try:
    import causal_conv1d; print('causal_conv1d: OK')
except ImportError:
    print('causal_conv1d: NOT AVAILABLE (will use fallback)')
"

echo ""
echo "=== Running Phase 0 Tests ==="
python tests/test_phase0.py
