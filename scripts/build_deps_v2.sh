#!/bin/bash
#SBATCH --job-name=build_deps2
#SBATCH --account=henderson
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/build_deps_v2_%j.out

module load anaconda3/2025.12
module load cudatoolkit/12.6
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
export CUDA_HOME=/usr/local/cuda-12.6

echo "Python: $(which python) $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA_HOME: $CUDA_HOME"
nvidia-smi --query-gpu=name --format=csv,noheader

echo ""
echo "=== Reinstalling flash-attn ==="
pip uninstall flash-attn -y 2>&1
pip install flash-attn --no-build-isolation --no-cache-dir 2>&1
echo "=== flash-attn done ==="

echo ""
echo "=== Installing mamba-ssm ==="
pip install mamba-ssm --no-build-isolation --no-cache-dir 2>&1
echo "=== mamba-ssm done ==="

echo ""
echo "=== Verification ==="
python -c "
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')

try:
    import flash_attn
    print('Flash Attention:', flash_attn.__version__)
except Exception as e:
    print('Flash Attention FAILED:', e)

try:
    import mamba_ssm
    print('Mamba SSM:', mamba_ssm.__version__)
except Exception as e:
    print('Mamba SSM FAILED:', e)

try:
    import causal_conv1d
    print('causal_conv1d: OK')
except Exception as e:
    print('causal_conv1d:', e)

import transformers; print('Transformers:', transformers.__version__)
import datasets; print('Datasets:', datasets.__version__)
import einops; print('Einops:', einops.__version__)
import wandb; print('WandB:', wandb.__version__)
print()
print('All imports successful!')
" 2>&1

echo ""
echo "=== Quick test: create a small Mamba2 model ==="
python -c "
import torch
from mamba_ssm.modules.mamba2 import Mamba2
model = Mamba2(d_model=128, d_state=64, d_conv=4, expand=2, headdim=32).cuda()
x = torch.randn(2, 32, 128).cuda()
y = model(x)
print('Mamba2 forward test:', y.shape)
print('SUCCESS')
" 2>&1

echo "=== All done ==="
