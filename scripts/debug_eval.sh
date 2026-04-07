#!/bin/bash
#SBATCH --job-name=debug_eval
#SBATCH --account=henderson
#SBATCH --time=15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/debug_eval_%j.out

module load anaconda3/2025.12
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
module load cudatoolkit/12.6
export CUDA_HOME=/usr/local/cuda-12.6

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

python3 << 'PYEOF'
import torch
import torch.nn.functional as F
import numpy as np
import math

from models.ssm_ttt_model import create_vanilla_ssm
from data.dataloader import get_tokenizer

device = "cuda"

# Load old v1 vanilla (1B tokens)
print("=" * 60)
print("CROSS-VALIDATION: old model vs new model on both datasets")
print("=" * 60)

print("\n--- Loading old vanilla (1B tokens) ---")
ckpt = torch.load("runs/tiny_pilot_vanilla_ssm/checkpoint_final.pt", map_location='cpu', weights_only=False)
config = ckpt["config"]
model_old = create_vanilla_ssm(**config["model_args"], device=device, dtype=torch.bfloat16)
model_old.load_state_dict(ckpt["model_state_dict"])
model_old.eval()

# Load val data
old_data = np.memmap("data_cache/val.bin", dtype=np.uint16, mode='r')
old_off = np.load("data_cache/val_offsets.npy")
new_data = np.memmap("data_cache_v2/val.bin", dtype=np.uint16, mode='r')
new_off = np.load("data_cache_v2/val_offsets.npy")

def eval_ppl_on_doc(model, doc_tokens, context_len=2048):
    """Evaluate PPL on last context_len tokens of a document."""
    if len(doc_tokens) < context_len:
        return None
    input_ids = torch.from_numpy(doc_tokens[-context_len:].astype(np.int64)).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(input_ids)
    logits = out.logits.float()
    # Score last 2048 tokens (all of them in this case)
    nll = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                          input_ids[:, 1:].reshape(-1), reduction='mean')
    return math.exp(nll.item())

def eval_ppl_on_dataset(model, data, offsets, n_docs=5, context_len=2048):
    ppls = []
    for i in range(min(n_docs, len(offsets)-1)):
        doc = data[offsets[i]:offsets[i+1]]
        if len(doc) >= context_len:
            ppl = eval_ppl_on_doc(model, doc, context_len)
            if ppl is not None:
                ppls.append(ppl)
    return sum(ppls) / len(ppls) if ppls else float('inf'), ppls

# Test old model on old data (should match ~7.33)
print("\n--- Old model (1B tok) on OLD val data (5 docs, 2k context) ---")
avg_ppl, ppls = eval_ppl_on_dataset(model_old, old_data, old_off, n_docs=5, context_len=2048)
print(f"  Average PPL: {avg_ppl:.2f}")
print(f"  Per-doc PPL: {[f'{p:.2f}' for p in ppls]}")

# Test old model on new data
print("\n--- Old model (1B tok) on NEW val data (5 docs, 2k context) ---")
avg_ppl, ppls = eval_ppl_on_dataset(model_old, new_data, new_off, n_docs=5, context_len=2048)
print(f"  Average PPL: {avg_ppl:.2f}")
print(f"  Per-doc PPL: {[f'{p:.2f}' for p in ppls]}")

# Test old model on new data (50 docs)
print("\n--- Old model (1B tok) on NEW val data (50 docs, 2k context) ---")
avg_ppl, ppls = eval_ppl_on_dataset(model_old, new_data, new_off, n_docs=50, context_len=2048)
print(f"  Average PPL: {avg_ppl:.2f}")
print(f"  Min/Max PPL: {min(ppls):.2f} / {max(ppls):.2f}")

del model_old
torch.cuda.empty_cache()

# Load new v3 vanilla (200M tokens)
print("\n--- Loading new v3 vanilla (200M tokens) ---")
ckpt = torch.load("runs/stage1_vanilla/checkpoint_final.pt", map_location='cpu', weights_only=False)
config = ckpt["config"]
model_new = create_vanilla_ssm(**config["model_args"], device=device, dtype=torch.bfloat16)
model_new.load_state_dict(ckpt["model_state_dict"])
model_new.eval()

# Test new model on old data
print("\n--- New model (200M tok) on OLD val data (5 docs, 2k context) ---")
avg_ppl, ppls = eval_ppl_on_dataset(model_new, old_data, old_off, n_docs=5, context_len=2048)
print(f"  Average PPL: {avg_ppl:.2f}")
print(f"  Per-doc PPL: {[f'{p:.2f}' for p in ppls]}")

# Test new model on new data
print("\n--- New model (200M tok) on NEW val data (5 docs, 2k context) ---")
avg_ppl, ppls = eval_ppl_on_dataset(model_new, new_data, new_off, n_docs=5, context_len=2048)
print(f"  Average PPL: {avg_ppl:.2f}")
print(f"  Per-doc PPL: {[f'{p:.2f}' for p in ppls]}")

# Test new model on new data (50 docs)
print("\n--- New model (200M tok) on NEW val data (50 docs, 2k context) ---")
avg_ppl, ppls = eval_ppl_on_dataset(model_new, new_data, new_off, n_docs=50, context_len=2048)
print(f"  Average PPL: {avg_ppl:.2f}")
print(f"  Min/Max PPL: {min(ppls):.2f} / {max(ppls):.2f}")

# Also check: what does the evaluate.py sliding window eval give for old model at 2k?
print("\n--- Checking evaluate.py's sliding-window logic ---")
from evaluate import load_val_documents, evaluate_sliding_window_ppl
docs_old = load_val_documents("data_cache", min_len=2048)
docs_new = load_val_documents("data_cache_v2", min_len=2048)
print(f"Old val docs >=2048: {len(docs_old)}")
print(f"New val docs >=2048: {len(docs_new)}")

del model_new
torch.cuda.empty_cache()

# Reload old model for the evaluate.py test
ckpt = torch.load("runs/tiny_pilot_vanilla_ssm/checkpoint_final.pt", map_location='cpu', weights_only=False)
config = ckpt["config"]
model_old = create_vanilla_ssm(**config["model_args"], device=device, dtype=torch.bfloat16)
model_old.load_state_dict(ckpt["model_state_dict"])
model_old.eval()

print("\n--- evaluate.py: old model on OLD val (5 docs), 2k only ---")
res, _ = evaluate_sliding_window_ppl(model_old, docs_old[:5], context_lengths=[2048], device=device)
print(f"  Result: {res}")

print("\n--- evaluate.py: old model on NEW val (5 docs), 2k only ---")
res, _ = evaluate_sliding_window_ppl(model_old, docs_new[:5], context_lengths=[2048], device=device)
print(f"  Result: {res}")

PYEOF
