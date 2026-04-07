#!/bin/bash
#SBATCH --job-name=debug_te
#SBATCH --account=henderson
#SBATCH --time=15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/debug_te_%j.out

module load anaconda3/2025.12
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
module load cudatoolkit/12.6
export CUDA_HOME=/usr/local/cuda-12.6

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

python3 -u << 'PYEOF'
import torch
import torch.nn.functional as F
import numpy as np
import math

from models.ssm_ttt_model import create_vanilla_ssm
from data.dataloader import get_tokenizer, DocOffsetTrainDataset

device = "cuda"

print("Loading v3 vanilla model (200M tokens)...")
ckpt = torch.load("runs/stage1_vanilla/checkpoint_final.pt", map_location='cpu', weights_only=False)
config = ckpt["config"]
model = create_vanilla_ssm(**config["model_args"], device=device, dtype=torch.bfloat16)
model.load_state_dict(ckpt["model_state_dict"])

train_data = np.memmap("data_cache_v2/train.bin", dtype=np.uint16, mode='r')
train_offsets = np.load("data_cache_v2/train_offsets.npy")

# Get a fixed batch
rng = np.random.default_rng(999)
valid = np.where((train_offsets[1:] - train_offsets[:-1]) >= 2048)[0]
batch_ids = []
for i in range(3):
    doc_idx = rng.choice(valid)
    doc_start = train_offsets[doc_idx]
    doc_end = train_offsets[doc_idx + 1]
    doc_len = doc_end - doc_start
    start = doc_start + rng.integers(0, doc_len - 2048 + 1)
    tokens = train_data[start:start+2048].astype(np.int64)
    batch_ids.append(torch.from_numpy(tokens))

input_ids = torch.stack(batch_ids).to(device)
labels = input_ids.clone()

# Test 1: model.eval() mode
print("\n=== model.eval() mode ===")
model.eval()
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(input_ids, labels=labels)
print(f"Loss: {out.loss.item():.4f}  PPL: {math.exp(out.loss.item()):.2f}")

# Test 2: model.train() mode
print("\n=== model.train() mode ===")
model.train()
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(input_ids, labels=labels)
print(f"Loss: {out.loss.item():.4f}  PPL: {math.exp(out.loss.item()):.2f}")

# Test 3: model.train() mode WITHOUT torch.no_grad (like actual training)
print("\n=== model.train() mode, WITH grad tracking ===")
model.train()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model(input_ids, labels=labels)
print(f"Loss: {out.loss.item():.4f}  PPL: {math.exp(out.loss.item()):.2f}")

# Test 4: Check if it's the Mamba conv state
# Mamba2 has a d_conv parameter; in training mode, convolution might use
# causal padding differently
print("\n=== Checking param norms ===")
total_params = 0
total_norm = 0.0
for name, p in model.named_parameters():
    total_params += p.numel()
    total_norm += p.norm().item() ** 2
print(f"Total params: {total_params:,}")
print(f"Total param L2 norm: {math.sqrt(total_norm):.4f}")

# Test 5: Also check - is there an issue with how the loss was computed at training time?
# Specifically, are labels all -100 anywhere?
print(f"\n=== Labels check ===")
print(f"Labels shape: {labels.shape}")
print(f"Labels min: {labels.min().item()}")
print(f"Labels max: {labels.max().item()}")
print(f"Any -100: {(labels == -100).any().item()}")

# Let me also check if there's maybe an older checkpoint saved at a different path
import os
run_dir = "runs/stage1_vanilla"
print(f"\n=== Files in {run_dir} ===")
for f in sorted(os.listdir(run_dir)):
    fpath = os.path.join(run_dir, f)
    if f.endswith('.pt'):
        sz = os.path.getsize(fpath)
        print(f"  {f}: {sz/1024/1024:.1f} MB")
    elif f.endswith('.jsonl'):
        print(f"  {f}")

# Test 6: Recreate the model from scratch (untrained) and check its loss
print("\n=== Untrained model loss ===")
model_fresh = create_vanilla_ssm(**config["model_args"], device=device, dtype=torch.bfloat16)
model_fresh.eval()
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_fresh = model_fresh(input_ids, labels=labels)
print(f"Untrained loss: {out_fresh.loss.item():.4f}  PPL: {math.exp(min(out_fresh.loss.item(), 20)):.2f}")

PYEOF
