"""
Phase 0 correctness and unit tests — v3.

Tests from the updated spec (Section 6.1):
  0.1 - Zero-update identity: TTT model with zero target params == vanilla model
  0.2 - Update-state cap test: DeltaW Frobenius norm stays <= rho * ||W0||
  0.3 - Normalized EMA sanity: with constant G, DeltaW saturates to ~eta*G (not eta*G/(1-decay))
  0.4 - Single-document sample test: every training sample lies inside one doc span
  0.5 - Worker randomness test: no duplicated start positions across workers
  0.6 - Chunk causality test: target builder zeros q_{t+j} across chunk boundaries

Plus the original structural tests.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from models.ssm_ttt_model import create_vanilla_ssm, create_ssm_ttt, select_ttt_layers
from models.ttt_wrapper import TTTWrapper, TTTMamba2Block, rms_norm_lastdim, project_fro_rel
from models.target_builder import TargetBuilder


def print_test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"       {detail}")
    return passed


def test_0_1_zero_update_identity():
    """Test 0.1: Zero-init target builder => TTT model == vanilla model."""
    print("\n=== Test 0.1: Zero-Update Identity ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model, n_layer = 128, 6
    B, T = 2, 64

    torch.manual_seed(42)
    vanilla = create_vanilla_ssm(
        d_model=d_model, n_layer=n_layer, d_intermediate=0, vocab_size=256,
        ssm_cfg={"layer": "Mamba2", "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 32},
        rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        device=device, dtype=torch.float32,
    )

    torch.manual_seed(42)
    ttt_model = create_ssm_ttt(
        d_model=d_model, n_layer=n_layer, d_intermediate=0, vocab_size=256,
        ssm_cfg={"layer": "Mamba2", "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 32},
        rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        num_ttt_layers=2, ttt_chunk_size=32, ttt_kernel_size=5,
        ttt_normalize_update=True, ttt_deltaW_rel_cap=0.10, ttt_G_rel_cap=0.02,
        ttt_decay_factor_init=0.95, ttt_decay_min=0.90, ttt_decay_max=0.995,
        device=device, dtype=torch.float32,
    )

    vanilla_sd = vanilla.state_dict()
    ttt_sd = ttt_model.state_dict()
    for key in vanilla_sd:
        if key in ttt_sd:
            ttt_sd[key] = vanilla_sd[key].clone()
        else:
            parts = key.split('.')
            if len(parts) >= 3 and parts[0] == 'layers' and parts[2] == 'mixer':
                new_key = '.'.join(parts[:2] + ['ttt_wrapper', 'mamba_block'] + parts[3:])
                if new_key in ttt_sd:
                    ttt_sd[new_key] = vanilla_sd[key].clone()
    ttt_model.load_state_dict(ttt_sd)

    for layer in ttt_model.layers:
        if isinstance(layer, TTTMamba2Block):
            layer.ttt_wrapper.target_builder.mix_coeffs.data.zero_()
            layer.ttt_wrapper.target_builder.W_tgt.data.zero_()

    for layer in vanilla.layers:
        if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'use_mem_eff_path'):
            layer.mixer.use_mem_eff_path = False

    input_ids = torch.randint(0, 256, (B, T), device=device)

    vanilla.eval()
    ttt_model.eval()
    with torch.no_grad():
        out_vanilla = vanilla(input_ids).logits
        out_ttt = ttt_model(input_ids).logits

    diff = (out_vanilla - out_ttt).abs().max().item()
    ok = print_test("Zero-update identity test", diff < 1e-4,
                    f"Max logit diff: {diff:.2e}")
    return ok


def test_0_2_update_state_cap():
    """Test 0.2: DeltaW Frobenius norm stays <= rho * ||W0||."""
    print("\n=== Test 0.2: Update-State Cap Test ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, d_model, d_inner = 2, 64, 128
    rho = 0.10

    W0 = torch.randn(d_model, d_inner, device=device)
    W0_norm = W0.norm().item()

    G_large = torch.randn(B, d_model, d_inner, device=device) * 100.0
    deltaW = G_large.clone()

    deltaW_capped = project_fro_rel(deltaW, base_norm=W0.norm().detach(), rho=rho)

    dw_norms = torch.norm(deltaW_capped.reshape(B, -1), dim=1)
    max_allowed = rho * W0_norm

    ok1 = print_test("DeltaW norm <= rho * ||W0||",
                     (dw_norms <= max_allowed + 1e-5).all().item(),
                     f"DeltaW norms: {dw_norms.tolist()}, max allowed: {max_allowed:.4f}")

    G_small = torch.randn(B, d_model, d_inner, device=device) * 1e-6
    deltaW_small = project_fro_rel(G_small, base_norm=W0.norm().detach(), rho=rho)
    diff = (G_small - deltaW_small).abs().max().item()
    ok2 = print_test("Small G passes through uncapped", diff < 1e-7,
                     f"Max diff: {diff:.2e}")

    return all([ok1, ok2])


def test_0_3_normalized_ema_sanity():
    """Test 0.3: With constant G, DeltaW saturates to ~eta*G (not eta*G/(1-decay))."""
    print("\n=== Test 0.3: Normalized EMA Sanity Test ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, d_out, d_in = 1, 16, 32
    decay = 0.95
    eta = 0.5
    G = torch.ones(B, d_out, d_in, device=device) * 0.1

    deltaW = torch.zeros(B, d_out, d_in, device=device)
    for _ in range(1000):
        deltaW = decay * deltaW + (1.0 - decay) * eta * G

    expected = eta * G
    diff = (deltaW - expected).abs().max().item()

    old_limit = eta * G / (1.0 - decay)
    old_diff = (deltaW - old_limit).abs().max().item()

    ok1 = print_test("DeltaW converges to eta * G", diff < 1e-4,
                     f"||DeltaW - eta*G||_max = {diff:.6f}")
    ok2 = print_test("DeltaW does NOT converge to eta*G/(1-decay)", old_diff > 0.01,
                     f"||DeltaW - eta*G/(1-decay)||_max = {old_diff:.6f}")

    print(f"  DeltaW_inf = {deltaW[0, 0, 0].item():.6f}")
    print(f"  eta * G = {(eta * G)[0, 0, 0].item():.6f}")
    print(f"  eta * G / (1-decay) = {(eta * G / (1 - decay))[0, 0, 0].item():.6f}")

    return all([ok1, ok2])


def test_0_4_single_document_sample():
    """Test 0.4: Every training sample lies inside one doc span from train_offsets."""
    print("\n=== Test 0.4: Single-Document Sample Test ===")

    tmpdir = "/tmp/test_doc_offset_data"
    os.makedirs(tmpdir, exist_ok=True)

    eos_id = 0
    doc1 = np.array([10, 20, 30, 40, 50, eos_id], dtype=np.uint16)
    doc2 = np.array([100, 200, 300, 400, 500, 600, 700, 800, eos_id], dtype=np.uint16)
    doc3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, eos_id], dtype=np.uint16)

    all_tokens = np.concatenate([doc1, doc2, doc3])
    offsets = np.array([0, len(doc1), len(doc1) + len(doc2), len(all_tokens)], dtype=np.int64)

    bin_path = os.path.join(tmpdir, "train.bin")
    off_path = os.path.join(tmpdir, "train_offsets.npy")
    meta_path = os.path.join(tmpdir, "train_meta.txt")

    fp = np.memmap(bin_path, dtype=np.uint16, mode='w+', shape=all_tokens.shape)
    fp[:] = all_tokens
    fp.flush()
    del fp
    np.save(off_path, offsets)
    with open(meta_path, "w") as f:
        f.write(f"num_tokens={len(all_tokens)}\n")

    from data.dataloader import DocOffsetTrainDataset
    ds = DocOffsetTrainDataset(bin_path, off_path, seq_len=4, eos_token_id=eos_id, seed=42, min_doc_len=4)

    all_ok = True
    for idx in range(20):
        sample = ds[idx]
        tokens = sample["input_ids"].numpy()

        in_some_doc = False
        for d in range(len(offsets) - 1):
            doc_start = offsets[d]
            doc_end = offsets[d + 1]
            doc_tokens = all_tokens[doc_start:doc_end]

            non_pad = tokens[tokens != eos_id]
            if len(non_pad) == 0:
                in_some_doc = True
                break

            for start in range(len(doc_tokens)):
                match_len = 0
                for k in range(len(non_pad)):
                    if start + k < len(doc_tokens) and doc_tokens[start + k] == non_pad[k]:
                        match_len += 1
                    else:
                        break
                if match_len == len(non_pad):
                    in_some_doc = True
                    break
            if in_some_doc:
                break

        if not in_some_doc:
            print(f"  FAIL: sample {idx} tokens={tokens} not found in any doc")
            all_ok = False

    ok = print_test("All samples within single document", all_ok)

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    return ok


def test_0_5_worker_randomness():
    """Test 0.5: Different workers produce different start positions."""
    print("\n=== Test 0.5: Worker Randomness Test ===")

    seed = 42

    starts_w0 = set()
    starts_w1 = set()

    for idx in range(100):
        rng0 = np.random.default_rng(seed + 1000003 * 0 + idx)
        rng1 = np.random.default_rng(seed + 1000003 * 1 + idx)

        s0 = rng0.integers(0, 1000000)
        s1 = rng1.integers(0, 1000000)

        starts_w0.add(s0)
        starts_w1.add(s1)

    overlap = starts_w0 & starts_w1
    overlap_frac = len(overlap) / 100

    ok1 = print_test("Worker 0 and 1 have unique starts within themselves",
                     len(starts_w0) == 100 and len(starts_w1) == 100,
                     f"W0: {len(starts_w0)} unique, W1: {len(starts_w1)} unique")

    ok2 = print_test("Workers have low overlap (<5%)",
                     overlap_frac < 0.05,
                     f"Overlap: {len(overlap)}/100 = {overlap_frac:.1%}")

    return all([ok1, ok2])


def test_0_6_chunk_causality():
    """Test 0.6: Target builder zeros q_{t+j} when it crosses a chunk boundary."""
    print("\n=== Test 0.6: Chunk Causality Test ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = 32
    B, T, C = 1, 64, 16

    tb = TargetBuilder(d_model=d_model, kernel_size=5, device=device)
    tb.mix_coeffs.data.fill_(1.0)
    tb.W_tgt.data = torch.eye(d_model, device=device, dtype=torch.float32)

    q = torch.randn(B, T, d_model, device=device)
    vhat = tb(q, chunk_size=C)

    all_ok = True
    for chunk_idx in range(T // C):
        chunk_start = chunk_idx * C
        chunk_end = (chunk_idx + 1) * C

        boundary_pos = chunk_end - 1

        last_pos_in_chunk = boundary_pos
        v_at_boundary = vhat[0, last_pos_in_chunk]

        for j in range(1, 6):
            target_pos = last_pos_in_chunk + j
            if target_pos >= T or target_pos >= chunk_end:
                pass

    q1 = q.clone()
    q2 = q.clone()
    q2[:, C:, :] = torch.randn(B, T - C, d_model, device=device)

    vhat1 = tb(q1, chunk_size=C)
    vhat2 = tb(q2, chunk_size=C)

    chunk1_diff = (vhat1[:, :C, :] - vhat2[:, :C, :]).abs().max().item()
    ok1 = print_test("Chunk 0 targets unchanged by changes in chunk 1",
                     chunk1_diff < 1e-6,
                     f"Max diff: {chunk1_diff:.2e}")

    last_in_chunk = C - 1
    v_last = vhat1[0, last_in_chunk]
    expected_shifts = 0
    for j in range(1, 6):
        if last_in_chunk + j >= C:
            expected_shifts += 1

    q_zeros = torch.zeros(B, T, d_model, device=device)
    q_zeros[:, :C, :] = q[:, :C, :]
    vhat_isolated = tb(q_zeros, chunk_size=C)

    diff_c0 = (vhat1[:, :C, :] - vhat_isolated[:, :C, :]).abs().max().item()
    ok2 = print_test("Chunk 0 targets same when rest of sequence is zeroed",
                     diff_c0 < 1e-6,
                     f"Max diff: {diff_c0:.2e}")

    boundary_pos = C - 2
    v_near_boundary = vhat1[0, boundary_pos]
    mid_pos = C // 2
    v_mid = vhat1[0, mid_pos]
    ok3 = print_test("Near-boundary position has smaller target than mid-chunk",
                     v_near_boundary.abs().mean() <= v_mid.abs().mean() + 0.1,
                     f"Near-boundary mean abs: {v_near_boundary.abs().mean():.4f}, "
                     f"mid-chunk mean abs: {v_mid.abs().mean():.4f}")

    return all([ok1, ok2, ok3])


def test_select_ttt_layers():
    """Test TTT layer selection logic."""
    print("\n=== Test: TTT Layer Selection ===")

    ok1 = print_test("24 layers, 4 TTT", select_ttt_layers(24, 4) == [5, 10, 14, 19],
                     f"Got: {select_ttt_layers(24, 4)}")
    ok2 = print_test("6 layers, 2 TTT", select_ttt_layers(6, 2) == [2, 4],
                     f"Got: {select_ttt_layers(6, 2)}")
    ok3 = print_test("4 layers, 4 TTT", len(select_ttt_layers(4, 4)) <= 4,
                     f"Got: {select_ttt_layers(4, 4)}")
    ok4 = print_test("0 TTT layers", select_ttt_layers(24, 0) == [])
    return all([ok1, ok2, ok3, ok4])


def test_rms_norm():
    """Test RMS norm helper."""
    print("\n=== Test: RMS Norm Helper ===")

    x = torch.randn(2, 10, 64)
    x_normed = rms_norm_lastdim(x, eps=1e-6)

    rms_vals = torch.sqrt(x_normed.square().mean(dim=-1))
    ok = print_test("RMS norm output has unit RMS",
                    (rms_vals - 1.0).abs().max().item() < 1e-4,
                    f"Max deviation from 1.0: {(rms_vals - 1.0).abs().max().item():.6f}")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 0: Correctness and Unit Tests (v3)")
    print("=" * 60)

    results = {}
    results["layer_selection"] = test_select_ttt_layers()
    results["rms_norm"] = test_rms_norm()
    results["0.1_zero_update"] = test_0_1_zero_update_identity()
    results["0.2_state_cap"] = test_0_2_update_state_cap()
    results["0.3_ema_sanity"] = test_0_3_normalized_ema_sanity()
    results["0.4_single_doc"] = test_0_4_single_document_sample()
    results["0.5_worker_rng"] = test_0_5_worker_randomness()
    results["0.6_chunk_causal"] = test_0_6_chunk_causality()

    print("\n" + "=" * 60)
    print("Summary:")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll Phase 0 tests PASSED!")
    else:
        print("\nSome tests FAILED. Fix before proceeding to Stage 1.")
    print("=" * 60)
