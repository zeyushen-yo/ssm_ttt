#!/bin/bash
#SBATCH --job-name=debug_loss
#SBATCH --account=henderson
#SBATCH --time=15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/debug_loss_%j.out

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

# Load v3 vanilla
print("Loading v3 vanilla model (200M tokens)...")
ckpt = torch.load("runs/stage1_vanilla/checkpoint_final.pt", map_location='cpu', weights_only=False)
config = ckpt["config"]
model = create_vanilla_ssm(**config["model_args"], device=device, dtype=torch.bfloat16)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 1. Compute loss on TRAINING data (same way as train.py)
print("\n=== Test 1: Loss on TRAINING data (like train.py) ===")
train_data = np.memmap("data_cache_v2/train.bin", dtype=np.uint16, mode='r')
train_offsets = np.load("data_cache_v2/train_offsets.npy")

losses_train = []
with torch.no_grad():
    for i in range(10):
        rng = np.random.default_rng(42 + i)
        valid = np.where((train_offsets[1:] - train_offsets[:-1]) >= 2048)[0]
        doc_idx = rng.choice(valid)
        doc_start = train_offsets[doc_idx]
        doc_end = train_offsets[doc_idx + 1]
        doc_len = doc_end - doc_start
        start = doc_start + rng.integers(0, doc_len - 2048 + 1)
        tokens = train_data[start:start+2048].astype(np.int64)
        input_ids = torch.from_numpy(tokens).unsqueeze(0).to(device)
        labels = input_ids.clone()
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(input_ids, labels=labels)
        loss = output.loss.item()
        losses_train.append(loss)
        print(f"  Train sample {i}: loss={loss:.4f} ppl={math.exp(loss):.2f}")

avg_train = sum(losses_train) / len(losses_train)
print(f"  AVERAGE: loss={avg_train:.4f} ppl={math.exp(avg_train):.2f}")

# 2. Compute loss on EVAL data (same way as evaluate.py)
print("\n=== Test 2: Loss on EVAL data (like evaluate.py) ===")
new_data = np.memmap("data_cache_v2/val.bin", dtype=np.uint16, mode='r')
new_off = np.load("data_cache_v2/val_offsets.npy")

losses_eval = []
with torch.no_grad():
    for i in range(10):
        doc = new_data[new_off[i]:new_off[i+1]]
        input_ids = torch.from_numpy(doc[-2048:].astype(np.int64)).unsqueeze(0).to(device)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(input_ids)
        logits = output.logits.float()
        
        # Same as evaluate.py for L=2048
        score_logits = logits[:, 0:2047, :]
        score_labels = input_ids[:, 1:2048]
        nll = F.cross_entropy(score_logits.reshape(-1, score_logits.size(-1)),
                              score_labels.reshape(-1), reduction='mean')
        ppl = math.exp(nll.item())
        losses_eval.append(nll.item())
        print(f"  Eval doc {i}: nll={nll.item():.4f} ppl={ppl:.2f}")

avg_eval = sum(losses_eval) / len(losses_eval)
print(f"  AVERAGE: nll={avg_eval:.4f} ppl={math.exp(avg_eval):.2f}")

# 3. Compute loss on eval data using model.forward(labels=) to match train
print("\n=== Test 3: Eval data but using model.forward(labels=) like train ===")
losses_eval2 = []
with torch.no_grad():
    for i in range(10):
        doc = new_data[new_off[i]:new_off[i+1]]
        input_ids = torch.from_numpy(doc[-2048:].astype(np.int64)).unsqueeze(0).to(device)
        labels = input_ids.clone()
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(input_ids, labels=labels)
        loss = output.loss.item()
        losses_eval2.append(loss)
        print(f"  Eval doc {i} (model.loss): loss={loss:.4f} ppl={math.exp(loss):.2f}")

avg_eval2 = sum(losses_eval2) / len(losses_eval2)
print(f"  AVERAGE: loss={avg_eval2:.4f} ppl={math.exp(avg_eval2):.2f}")

# 4. Check if autocast makes a difference
print("\n=== Test 4: Eval WITHOUT autocast ===")
losses_nocast = []
with torch.no_grad():
    for i in range(3):
        doc = new_data[new_off[i]:new_off[i+1]]
        input_ids = torch.from_numpy(doc[-2048:].astype(np.int64)).unsqueeze(0).to(device)
        
        output = model(input_ids)  # NO autocast
        logits = output.logits.float()
        nll = F.cross_entropy(logits[:, 0:2047, :].reshape(-1, logits.size(-1)),
                              input_ids[:, 1:2048].reshape(-1), reduction='mean')
        losses_nocast.append(nll.item())
        print(f"  No-autocast doc {i}: nll={nll.item():.4f} ppl={math.exp(nll.item()):.2f}")

print(f"\n=== SUMMARY ===")
print(f"Train data avg:         loss={avg_train:.4f} ppl={math.exp(avg_train):.2f}")
print(f"Eval data (manual CE):  nll={avg_eval:.4f} ppl={math.exp(avg_eval):.2f}")
print(f"Eval data (model.loss): loss={avg_eval2:.4f} ppl={math.exp(avg_eval2):.2f}")
print(f"Gap (eval/train):       {math.exp(avg_eval)/math.exp(avg_train):.1f}x")

PYEOF
