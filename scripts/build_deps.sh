#!/bin/bash
#SBATCH --job-name=build_deps
#SBATCH --account=henderson
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/build_deps_%j.out

module load anaconda3/2025.12
module load cudatoolkit/12.6
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
export CUDA_HOME=/usr/local/cuda-12.6

echo "=== Building causal-conv1d ==="
pip install causal-conv1d --no-build-isolation --no-cache-dir 2>&1
echo "=== causal-conv1d done ==="

echo "=== Verifying mamba-ssm ==="
python -c "import mamba_ssm; print('mamba_ssm version:', mamba_ssm.__version__)" 2>&1

echo "=== Verifying all packages ==="
python -c "
import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
import transformers; print('Transformers:', transformers.__version__)
import flash_attn; print('Flash Attention:', flash_attn.__version__)
import mamba_ssm; print('Mamba SSM:', mamba_ssm.__version__)
try:
    import causal_conv1d; print('causal_conv1d: OK')
except ImportError:
    print('causal_conv1d: NOT AVAILABLE')
import datasets; print('Datasets:', datasets.__version__)
import einops; print('Einops:', einops.__version__)
import wandb; print('WandB:', wandb.__version__)
" 2>&1

echo "=== All done ==="
