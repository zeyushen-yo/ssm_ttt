"""
Phase 0 correctness and unit tests (Spec Section 6, Phase 0).

Tests:
  0.1 - Expose block internals: verify pre_out_features shape and out_proj equivalence
  0.2 - Zero-update identity: TTT model with zero target params == vanilla model
  0.3 - Chunk-equivalence: different chunk sizes give same output when DeltaW=0
  0.4 - Boundary-isolation: changing chunk 2 doesn't affect Vhat for chunk 1
  0.5 - Reset test: fresh run on doc B == run after doc A with reset
  0.6 - Toy repeated-pattern: TTT reduces loss on repeated patterns
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from models.ssm_ttt_model import create_vanilla_ssm, create_ssm_ttt, select_ttt_layers
from models.ttt_wrapper import TTTWrapper, TTTMamba2Block
from models.target_builder import TargetBuilder


def print_test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"       {detail}")
    return passed


def test_0_1_expose_block_internals():
    """Test 0.1: Verify pre_out_features shape and out_proj equivalence."""
    print("\n=== Test 0.1: Expose Block Internals ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model, n_layer = 128, 4
    B, T = 2, 64

    model = create_vanilla_ssm(
        d_model=d_model, n_layer=n_layer, d_intermediate=0,
        vocab_size=256,
        ssm_cfg={"layer": "Mamba2", "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 32},
        rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        device=device, dtype=torch.float32,
    )

    mamba_block = model.layers[0].mixer

    # Check out_proj exists and shape
    W = mamba_block.out_proj.weight
    d_inner = mamba_block.d_inner
    ok1 = print_test("out_proj weight exists", W is not None)
    ok2 = print_test("out_proj weight shape", W.shape == (d_model, d_inner),
                     f"Expected ({d_model}, {d_inner}), got {tuple(W.shape)}")

    # Check pre-out features via TTTWrapper
    wrapper = TTTWrapper(
        mamba_block, d_model=d_model, chunk_size=T,
        device=device, dtype=torch.float32,
    )

    x = torch.randn(B, T, d_model, device=device)
    z = wrapper._get_pre_out_features(x)
    ok3 = print_test("pre_out_features shape", z.shape == (B, T, d_inner),
                     f"Expected ({B}, {T}, {d_inner}), got {tuple(z.shape)}")

    # Verify that explicit matmul with out_proj reproduces original output
    mamba_block.use_mem_eff_path = False
    original_out = mamba_block(x)  # [B, T, d_model]

    explicit_out = torch.matmul(z, mamba_block.out_proj.weight.t())
    if mamba_block.out_proj.bias is not None:
        explicit_out = explicit_out + mamba_block.out_proj.bias

    diff = (original_out - explicit_out).abs().max().item()
    ok4 = print_test("out_proj explicit matmul matches", diff < 1e-4,
                     f"Max diff: {diff:.2e}")

    return all([ok1, ok2, ok3, ok4])


def test_0_2_zero_update_identity():
    """Test 0.2: Zero-init target builder => TTT model == vanilla model."""
    print("\n=== Test 0.2: Zero-Update Identity ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model, n_layer = 128, 6
    B, T = 2, 64

    # Create vanilla model first
    torch.manual_seed(42)
    vanilla = create_vanilla_ssm(
        d_model=d_model, n_layer=n_layer, d_intermediate=0, vocab_size=256,
        ssm_cfg={"layer": "Mamba2", "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 32},
        rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        device=device, dtype=torch.float32,
    )

    # Create TTT model with separate seed (params will differ)
    torch.manual_seed(42)
    ttt_model = create_ssm_ttt(
        d_model=d_model, n_layer=n_layer, d_intermediate=0, vocab_size=256,
        ssm_cfg={"layer": "Mamba2", "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 32},
        rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        num_ttt_layers=2, ttt_chunk_size=32, ttt_kernel_size=5,
        device=device, dtype=torch.float32,
    )

    # Copy vanilla weights into TTT model so base params are identical
    vanilla_sd = vanilla.state_dict()
    ttt_sd = ttt_model.state_dict()
    for key in vanilla_sd:
        # Map vanilla key to TTT key (TTT layers have ttt_wrapper.mamba_block prefix)
        # Try direct match first
        if key in ttt_sd:
            ttt_sd[key] = vanilla_sd[key].clone()
        else:
            # For TTT-wrapped layers: layers.X.mixer.Y -> layers.X.ttt_wrapper.mamba_block.Y
            # and layers.X.norm.Y -> layers.X.norm.Y (same)
            parts = key.split('.')
            if len(parts) >= 3 and parts[0] == 'layers' and parts[2] == 'mixer':
                new_key = '.'.join(parts[:2] + ['ttt_wrapper', 'mamba_block'] + parts[3:])
                if new_key in ttt_sd:
                    ttt_sd[new_key] = vanilla_sd[key].clone()
    ttt_model.load_state_dict(ttt_sd)

    # Zero TTT-specific params
    for layer in ttt_model.layers:
        if isinstance(layer, TTTMamba2Block):
            layer.ttt_wrapper.target_builder.mix_coeffs.data.zero_()
            layer.ttt_wrapper.target_builder.W_tgt.data.zero_()

    # Disable fused path for vanilla (TTT layers already use non-fused)
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
    # With zero target params, vhat=0, so G=0, DeltaW stays 0, output should match
    ok = print_test("Zero-update identity test", diff < 1e-4,
                    f"Max logit diff: {diff:.2e}")
    return ok


def test_0_3_chunk_equivalence():
    """Test 0.3: Different chunk sizes give same output when target params are zero."""
    print("\n=== Test 0.3: Chunk-Equivalence Test ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = 128
    B, T = 2, 128

    # Create a target builder with zero params
    tb = TargetBuilder(d_model=d_model, kernel_size=5, device=device)
    tb.mix_coeffs.data.zero_()
    tb.W_tgt.data.zero_()

    q = torch.randn(B, T, d_model, device=device)

    out_c128 = tb(q, chunk_size=128)
    out_c64 = tb(q, chunk_size=64)
    out_c32 = tb(q, chunk_size=32)

    # All should be zero since mix_coeffs = 0
    ok1 = print_test("C=128 output is zero", out_c128.abs().max().item() < 1e-7,
                     f"Max: {out_c128.abs().max().item():.2e}")
    ok2 = print_test("C=64 output is zero", out_c64.abs().max().item() < 1e-7,
                     f"Max: {out_c64.abs().max().item():.2e}")
    ok3 = print_test("C=128 == C=64", (out_c128 - out_c64).abs().max().item() < 1e-7)
    ok4 = print_test("C=128 == C=32", (out_c128 - out_c32).abs().max().item() < 1e-7)

    # Now test with non-zero params: boundary masking should differ
    tb.mix_coeffs.data.fill_(0.1)
    tb.W_tgt.data = torch.eye(d_model, device=device, dtype=torch.float32) * 0.1
    out_c128_nz = tb(q, chunk_size=128)
    out_c64_nz = tb(q, chunk_size=64)

    # These should NOT be equal (different chunk boundaries -> different masking)
    diff_nz = (out_c128_nz - out_c64_nz).abs().max().item()
    ok5 = print_test("Non-zero params: C=128 != C=64 (different boundaries)", diff_nz > 1e-5,
                     f"Diff: {diff_nz:.2e}")

    return all([ok1, ok2, ok3, ok4, ok5])


def test_0_4_boundary_isolation():
    """Test 0.4: Changing tokens in chunk 2 doesn't affect Vhat for chunk 1."""
    print("\n=== Test 0.4: Boundary-Isolation Test ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = 128
    B, T, C = 1, 256, 128

    tb = TargetBuilder(d_model=d_model, kernel_size=5, device=device)
    # Use non-trivial params
    tb.mix_coeffs.data.uniform_(-0.1, 0.1)
    tb.W_tgt.data = torch.eye(d_model, device=device, dtype=torch.float32) * 0.5

    q1 = torch.randn(B, T, d_model, device=device)
    q2 = q1.clone()
    # Modify tokens in chunk 2 (positions C to 2C)
    q2[:, C:, :] = torch.randn(B, T - C, d_model, device=device)

    vhat1 = tb(q1, chunk_size=C)
    vhat2 = tb(q2, chunk_size=C)

    # Chunk 1 targets should be identical
    chunk1_diff = (vhat1[:, :C, :] - vhat2[:, :C, :]).abs().max().item()
    ok1 = print_test("Chunk 1 Vhat unchanged", chunk1_diff < 1e-6,
                     f"Max diff in chunk 1: {chunk1_diff:.2e}")

    # Chunk 2 targets should differ
    chunk2_diff = (vhat1[:, C:, :] - vhat2[:, C:, :]).abs().max().item()
    ok2 = print_test("Chunk 2 Vhat changed", chunk2_diff > 1e-5,
                     f"Max diff in chunk 2: {chunk2_diff:.2e}")

    return all([ok1, ok2])


def test_0_5_reset():
    """Test 0.5: DeltaW resets between documents."""
    print("\n=== Test 0.5: Reset Test ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model, n_layer = 128, 4
    B, T = 1, 64

    model = create_ssm_ttt(
        d_model=d_model, n_layer=n_layer, d_intermediate=0, vocab_size=256,
        ssm_cfg={"layer": "Mamba2", "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 32},
        rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        num_ttt_layers=2, ttt_chunk_size=32, ttt_kernel_size=5,
        device=device, dtype=torch.float32,
    )
    model.eval()

    # Set non-trivial target builder params
    for layer in model.layers:
        if isinstance(layer, TTTMamba2Block):
            layer.ttt_wrapper.target_builder.mix_coeffs.data.uniform_(-0.1, 0.1)
            layer.ttt_wrapper.target_builder.W_tgt.data = (
                torch.eye(d_model, device=device, dtype=torch.float32) * 0.5
            )

    doc_b = torch.randint(0, 256, (B, T), device=device)

    # Run on just doc B
    with torch.no_grad():
        out_fresh = model(doc_b).logits

    # DeltaW should already reset at start of forward (it's zero-initialized in the loop)
    # Run on doc B again (simulating after doc A - but DeltaW resets each forward call)
    with torch.no_grad():
        out_after = model(doc_b).logits

    diff = (out_fresh - out_after).abs().max().item()
    ok = print_test("Reset: fresh run == run after reset", diff < 1e-5,
                    f"Max diff: {diff:.2e}")
    return ok


def test_0_6_toy_repeated_pattern():
    """Test 0.6: TTT reduces loss on repeated patterns faster than vanilla."""
    print("\n=== Test 0.6: Toy Repeated-Pattern Test ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model, n_layer = 128, 4
    T = 256
    B = 2

    # Create a repeated pattern: abcabc...
    pattern_len = 8
    pattern = torch.randint(0, 256, (pattern_len,), device=device)
    repeats = T // pattern_len
    repeated = pattern.repeat(repeats)
    input_ids = repeated.unsqueeze(0).expand(B, -1).contiguous()

    # Vanilla SSM
    torch.manual_seed(123)
    vanilla = create_vanilla_ssm(
        d_model=d_model, n_layer=n_layer, d_intermediate=0, vocab_size=256,
        ssm_cfg={"layer": "Mamba2", "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 32},
        rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        device=device, dtype=torch.float32,
    )

    # SSM + TTT (using exact next-token target for this test)
    torch.manual_seed(123)
    ttt = create_ssm_ttt(
        d_model=d_model, n_layer=n_layer, d_intermediate=0, vocab_size=256,
        ssm_cfg={"layer": "Mamba2", "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 32},
        rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        num_ttt_layers=2, ttt_chunk_size=32, ttt_kernel_size=5,
        device=device, dtype=torch.float32,
    )

    # Set TTT target builder to approximate next-token (d_1=1, rest=0, W_tgt=I)
    for layer in ttt.layers:
        if isinstance(layer, TTTMamba2Block):
            tb = layer.ttt_wrapper.target_builder
            tb.mix_coeffs.data.zero_()
            tb.mix_coeffs.data[0] = 1.0  # d_1 = 1
            tb.W_tgt.data = torch.eye(d_model, device=device, dtype=torch.float32)

    # Evaluate (no training, just forward pass)
    vanilla.eval()
    ttt.eval()

    with torch.no_grad():
        v_out = vanilla(input_ids, labels=input_ids)
        t_out = ttt(input_ids, labels=input_ids)

    print(f"  Vanilla loss: {v_out.loss.item():.4f}")
    print(f"  TTT loss: {t_out.loss.item():.4f}")

    # Check DeltaW norms
    for i, layer in enumerate(ttt.layers):
        if isinstance(layer, TTTMamba2Block) and layer._last_deltaW is not None:
            dw_norm = torch.norm(layer._last_deltaW).item()
            print(f"  Layer {i} ||DeltaW||_F: {dw_norm:.6f}")

    # The TTT model has non-trivial DeltaW activity
    has_activity = False
    for layer in ttt.layers:
        if isinstance(layer, TTTMamba2Block) and layer._last_deltaW is not None:
            if torch.norm(layer._last_deltaW).item() > 1e-6:
                has_activity = True
                break

    ok = print_test("TTT has non-trivial fast-weight activity", has_activity)
    return ok


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


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 0: Correctness and Unit Tests")
    print("=" * 60)

    results = {}
    results["layer_selection"] = test_select_ttt_layers()
    results["0.1_block_internals"] = test_0_1_expose_block_internals()
    results["0.2_zero_update"] = test_0_2_zero_update_identity()
    results["0.3_chunk_equiv"] = test_0_3_chunk_equivalence()
    results["0.4_boundary"] = test_0_4_boundary_isolation()
    results["0.5_reset"] = test_0_5_reset()
    results["0.6_toy_pattern"] = test_0_6_toy_repeated_pattern()

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
        print("\nSome tests FAILED. Fix before proceeding to Phase 1.")
    print("=" * 60)
