#!/bin/bash
#SBATCH --job-name=test_32k
#SBATCH --account=henderson
#SBATCH --time=10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/HENDERSON/zs7353/ssm_ttt/scripts/test_32k_%j.out

module load anaconda3/2025.12
conda activate /scratch/gpfs/HENDERSON/zs7353/envs/ssm_ttt
module load cudatoolkit/12.6
export CUDA_HOME=/usr/local/cuda-12.6

cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt

python3 -u << 'PYEOF'
import torch
import torch.nn as nn

SEQ_LEN = 32768
BATCH = 1
DEVICE = "cuda"
DTYPE = torch.bfloat16

print(f"GPU: {torch.cuda.get_device_name()}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Testing seq_len={SEQ_LEN}, batch_size={BATCH}")

# Test 1: SWA Transformer
print("\n=== SWA Transformer ===")
try:
    from models.swa_transformer import SWATransformerLM
    model = SWATransformerLM(d_model=768, n_layer=12, d_ff=2048, num_heads=12,
                              window_size=512, vocab_size=50277, device=DEVICE, dtype=DTYPE)
    input_ids = torch.randint(0, 50277, (BATCH, SEQ_LEN), device=DEVICE)
    labels = input_ids.clone()
    
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=DTYPE):
        out = model(input_ids, labels=labels)
        loss = out.loss
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Forward+backward OK! Peak memory: {peak:.2f} GB, Loss: {loss.item():.4f}")
    del model, out, loss, input_ids, labels
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  FAILED: {e}")
    torch.cuda.empty_cache()

# Test 2: Vanilla SSM
print("\n=== Vanilla SSM ===")
try:
    from models.ssm_ttt_model import create_vanilla_ssm
    model = create_vanilla_ssm(d_model=768, n_layer=24, d_intermediate=0,
        ssm_cfg={"layer":"Mamba2","d_state":128,"d_conv":4,"expand":2,"headdim":64},
        rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        vocab_size=50277, device=DEVICE, dtype=DTYPE)
    input_ids = torch.randint(0, 50277, (BATCH, SEQ_LEN), device=DEVICE)
    labels = input_ids.clone()
    
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=DTYPE):
        out = model(input_ids, labels=labels)
        loss = out.loss
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Forward+backward OK! Peak memory: {peak:.2f} GB, Loss: {loss.item():.4f}")
    del model, out, loss, input_ids, labels
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  FAILED: {e}")
    torch.cuda.empty_cache()

# Test 3: SSM+TTT (C1 config: chunk=128, 4 layers)
print("\n=== SSM+TTT (chunk=128, 4 layers) ===")
try:
    from models.ssm_ttt_model import create_ssm_ttt
    model = create_ssm_ttt(d_model=768, n_layer=24, d_intermediate=0,
        ssm_cfg={"layer":"Mamba2","d_state":128,"d_conv":4,"expand":2,"headdim":64},
        rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        vocab_size=50277, num_ttt_layers=4, ttt_chunk_size=128, ttt_kernel_size=5,
        ttt_target_source="embedding", ttt_detach_source=False,
        ttt_inner_lr_init=0.10, ttt_decay_factor_init=0.95,
        ttt_decay_min=0.90, ttt_decay_max=0.995,
        ttt_normalize_update=True, ttt_norm_eps=1e-6,
        ttt_deltaW_rel_cap=0.10, ttt_G_rel_cap=0.02,
        device=DEVICE, dtype=DTYPE)
    input_ids = torch.randint(0, 50277, (BATCH, SEQ_LEN), device=DEVICE)
    labels = input_ids.clone()
    
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=DTYPE):
        out = model(input_ids, labels=labels)
        loss = out.loss
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Forward+backward OK! Peak memory: {peak:.2f} GB, Loss: {loss.item():.4f}")
    del model, out, loss, input_ids, labels
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  FAILED: {e}")
    torch.cuda.empty_cache()

# Test 4: SSM+TTT (C2 config: chunk=64, 4 layers)
print("\n=== SSM+TTT (chunk=64, 4 layers) ===")
try:
    from models.ssm_ttt_model import create_ssm_ttt
    model = create_ssm_ttt(d_model=768, n_layer=24, d_intermediate=0,
        ssm_cfg={"layer":"Mamba2","d_state":128,"d_conv":4,"expand":2,"headdim":64},
        rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        vocab_size=50277, num_ttt_layers=4, ttt_chunk_size=64, ttt_kernel_size=5,
        ttt_target_source="embedding", ttt_detach_source=False,
        ttt_inner_lr_init=0.10, ttt_decay_factor_init=0.95,
        ttt_decay_min=0.90, ttt_decay_max=0.995,
        ttt_normalize_update=True, ttt_norm_eps=1e-6,
        ttt_deltaW_rel_cap=0.10, ttt_G_rel_cap=0.02,
        device=DEVICE, dtype=DTYPE)
    input_ids = torch.randint(0, 50277, (BATCH, SEQ_LEN), device=DEVICE)
    labels = input_ids.clone()
    
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=DTYPE):
        out = model(input_ids, labels=labels)
        loss = out.loss
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Forward+backward OK! Peak memory: {peak:.2f} GB, Loss: {loss.item():.4f}")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nDone!")
PYEOF
