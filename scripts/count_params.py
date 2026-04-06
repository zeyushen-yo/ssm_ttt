"""
Utility to count and match parameters across the three model architectures.

Spec section 8: Parameters must match within ±5%.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.swa_transformer import SWATransformerLM


def count_transformer_params(d_model, n_layer, d_ff, num_heads, vocab_size=50280):
    """Count parameters in SWA Transformer."""
    model = SWATransformerLM(
        d_model=d_model, n_layer=n_layer, d_ff=d_ff,
        num_heads=num_heads, vocab_size=vocab_size,
        device="meta",
    )
    total = sum(p.numel() for p in model.parameters())
    # Subtract tied embeddings (counted once)
    emb = vocab_size * d_model
    # Note: with weight tying, lm_head.weight IS embedding.weight, so total already correct
    return total, model


def count_ssm_params(d_model, n_layer, d_intermediate, vocab_size=50280,
                     ssm_cfg=None, num_ttt_layers=0, **kwargs):
    """Count parameters in SSM model."""
    from models.ssm_ttt_model import create_vanilla_ssm, create_ssm_ttt
    if num_ttt_layers > 0:
        model = create_ssm_ttt(
            d_model=d_model, n_layer=n_layer, d_intermediate=d_intermediate,
            vocab_size=vocab_size, ssm_cfg=ssm_cfg,
            num_ttt_layers=num_ttt_layers, device="meta", **kwargs,
        )
    else:
        model = create_vanilla_ssm(
            d_model=d_model, n_layer=n_layer, d_intermediate=d_intermediate,
            vocab_size=vocab_size, ssm_cfg=ssm_cfg, device="meta", **kwargs,
        )
    total = sum(p.numel() for p in model.parameters())
    return total, model


def find_matched_configs():
    """Find model configurations that match parameter counts."""
    vocab_size = 50280

    print("=" * 70)
    print("Parameter Matching for Phase 1 (Tiny Pilot ~125M)")
    print("=" * 70)

    # Transformer-SWA config
    t_d, t_n, t_ff, t_h = 768, 12, 2048, 12
    t_params, _ = count_transformer_params(t_d, t_n, t_ff, t_h, vocab_size)
    print(f"\nTransformer-SWA: d={t_d}, n_layer={t_n}, d_ff={t_ff}, heads={t_h}")
    print(f"  Total params: {t_params:,}")

    # Vanilla SSM configs to try
    print("\nSearching SSM configs to match...")
    ssm_base = {"layer": "Mamba2", "d_state": 128, "d_conv": 4, "expand": 2, "headdim": 64}

    for d in [512, 640, 768, 896, 1024]:
        for n in [12, 16, 20, 24, 28, 32]:
            for d_int in [0]:
                try:
                    s_params, _ = count_ssm_params(d, n, d_int, vocab_size, ssm_base)
                    ratio = s_params / t_params
                    if 0.95 <= ratio <= 1.05:
                        print(f"  MATCH: d={d}, n_layer={n}, d_int={d_int} -> {s_params:,} "
                              f"(ratio: {ratio:.3f})")
                except Exception:
                    pass

    print("\n" + "=" * 70)
    print("Parameter Matching for Phase 3 (Main ~500M)")
    print("=" * 70)

    # Transformer-SWA: paper's 500M config
    t_d, t_n, t_ff, t_h = 1024, 24, 3072, 8
    t_params, _ = count_transformer_params(t_d, t_n, t_ff, t_h, vocab_size)
    print(f"\nTransformer-SWA: d={t_d}, n_layer={t_n}, d_ff={t_ff}, heads={t_h}")
    print(f"  Total params: {t_params:,}")

    print("\nSearching SSM configs to match...")
    for d in [768, 896, 1024, 1152, 1280]:
        for n in [24, 32, 36, 40, 48]:
            for d_int in [0]:
                try:
                    s_params, _ = count_ssm_params(d, n, d_int, vocab_size, ssm_base)
                    ratio = s_params / t_params
                    if 0.95 <= ratio <= 1.05:
                        print(f"  MATCH: d={d}, n_layer={n}, d_int={d_int} -> {s_params:,} "
                              f"(ratio: {ratio:.3f})")
                except Exception:
                    pass

    # Also count TTT overhead for matched SSM config
    print("\n--- TTT overhead estimate ---")
    for d in [768, 1024]:
        n = 24
        ttt_extra = 4 * (5 * d + d * d)  # 4 layers * (K*d mix_coeffs + d*d W_tgt)
        print(f"  d={d}, 4 TTT layers: extra = {ttt_extra:,} params")


if __name__ == "__main__":
    find_matched_configs()
