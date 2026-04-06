# SSM + In-Place TTT: Phase 1 Report

## 1. What Was Implemented (from the proposal)

### Fully implemented

- **Three matched-parameter models** (~131M params each, within ±2%):
  - **Vanilla SSM**: 24-layer Mamba2, d_model=768, expand=2, d_state=128
  - **SWA Transformer**: 12-layer decoder-only Transformer with sliding-window attention (window=512), RoPE, SwiGLU FFN, Flash Attention 2
  - **SSM + TTT**: Same Mamba2 backbone with 4 TTT-wrapped layers at indices [5, 10, 14, 19]

- **TTT mechanism** (spec sections 4.3–4.10, 5.1–5.6):
  - Intercepts Mamba2 `out_proj` to apply `(W_0 + DeltaW) @ Z` chunkwise
  - LM-aligned target builder with learnable `mix_coeffs` (kernel_size=5) and `W_tgt` projection
  - Apply-then-update loop: compute output with current DeltaW, then update DeltaW
  - Source embeddings = token embeddings (detached), per spec
  - DeltaW in fp32, reset at document boundaries
  - Chunk size = 128

- **TTT v2 enhancements** (beyond the original spec, added to fix DeltaW explosion):
  - Learnable per-layer inner learning rate (`eta`, initialized at 0.01, stored in log-space)
  - Learnable per-layer EMA decay factor (`decay`, initialized at 0.95, stored in logit-space)
  - Gradient clipping on G with `clip_tau=1.0`
  - Update rule: `G = eta * (1/C) * Vhat^T @ Z`, clip G, then `DeltaW = decay * DeltaW + G`

- **Data pipeline**:
  - Pre-tokenized The Pile (`monology/pile-uncopyrighted`) using GPT-NeoX tokenizer
  - ~2B tokens in `train.bin` (memory-mapped), validation documents with per-doc offsets
  - Incremental writing to handle memory constraints

- **Training infrastructure**:
  - YAML-configurable training script with AdamW, cosine LR decay, bfloat16 autocast
  - Checkpoint resuming (auto-finds latest checkpoint)
  - Offline WandB logging
  - TTT diagnostics: DeltaW norms, inner_lr, decay_factor per layer per log step

- **Figure-2 evaluation** (spec section 7):
  - Sliding-window perplexity at context lengths {2k, 4k, 8k, 16k, 32k}
  - Fixed 2048-token scored suffix
  - Comparison plot generation

- **Phase 0 unit tests**: forward pass, parameter counting, TTT zero-update identity test — all passed

### Not yet implemented

- **Phase 2** (medium pilot: 300–500M params, 2–5B tokens)
- **Phase 3** (main deliverable: 500M params, 20B tokens, seq_len=32768)
- **Document boundary resets during evaluation** (not needed for Phase 1 since eval docs are single documents)
- **Throughput/memory benchmarking** (spec section 9.2)
- **Toy repeated-pattern test** (spec TODO 0.6)
- **Chunk-equivalence test** (spec TODO 0.3)

---

## 2. Phase 1 Results

### 2.1 Training Configuration

| | Vanilla SSM | SWA Transformer | SSM + TTT v2 |
|---|---|---|---|
| Parameters | 131.0M | 131.2M | 131.4M (2.4M TTT extra) |
| Layers | 24 | 12 | 24 (4 TTT-wrapped) |
| Training tokens | 1B | 1B | 1B |
| Sequence length | 2048 | 2048 | 2048 |
| Batch size | 16 | 16 | 16 |
| Peak LR | 6e-4 | 6e-4 | 6e-4 |
| Final training loss | 3.38 | 3.42 | **3.19** |

### 2.2 Sliding-Window Perplexity (lower is better)

Evaluated on 5 validation documents from The Pile (≥32k tokens each), scoring the last 2048 tokens.

| Model | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|
| Vanilla SSM | 7.33 | 6.86 | 6.92 | 6.98 | **7.02** |
| SWA Transformer | **6.44** | 6.88 | 9.08 | 16.18 | 32.10 |
| SSM + TTT v1 (original, broken) | 6.76 | 8.05 | 57.74 | 417.87 | 1227.13 |
| **SSM + TTT v2 (with fixes)** | 6.80 | **6.15** | 7.99 | 20.77 | 76.32 |

### 2.3 Key Observations

1. **TTT v2 beats all baselines at 4k context** (PPL 6.15 vs 6.44 for SWA, 6.86 for vanilla SSM). This is a genuine positive signal.

2. **TTT v2 also beats SWA at 8k** (7.99 vs 9.08), though it falls behind vanilla SSM (6.92).

3. **Vanilla SSM is remarkably stable** across all context lengths (~7.0 PPL), making it the best at 8k+.

4. **SWA Transformer degrades at long contexts** — this is expected behavior when trained on 2048-token sequences, not a bug. With 12 layers × 512-token window, the effective receptive field is ~6k tokens. At eval, the residual stream statistics at positions far from any sequence boundary differ from training.

5. **TTT v2 is a massive improvement over v1** — at 8k, 7.99 vs 57.74; at 32k, 76 vs 1227.

### 2.4 Learned TTT Parameters (converged)

| Layer | inner_lr (η) | decay_factor | Effective memory window |
|---|---|---|---|
| 5 | 0.366 | 0.986 | 71 chunks ≈ 9k tokens |
| 10 | 0.323 | 0.997 | 333 chunks ≈ 43k tokens |
| 14 | 0.323 | 0.996 | 250 chunks ≈ 32k tokens |
| 19 | 0.385 | 0.904 | 10 chunks ≈ 1.3k tokens |

The model learned qualitatively different forgetting rates per layer: deeper layers (19) forget quickly for fast adaptation, while middle layers (10, 14) retain information over very long windows.

---

## 3. Analysis: Why TTT Works at 4k but Degrades at 16k+

### Root Cause: Extrapolation Gap

The DeltaW fast weights accumulate as an exponential moving average:

```
DeltaW_n = decay * DeltaW_{n-1} + G_n
```

The effective DeltaW magnitude after `n` chunks is proportional to `(1 - decay^n) / (1 - decay)`. The ratio between eval and training DeltaW determines the extrapolation stress:

| Context | Chunks | Layer 10 ratio | Layer 19 ratio | TTT PPL |
|---|---|---|---|---|
| 2k | 16 | 1.0× | 1.0× | 6.80 |
| 4k | 32 | 2.0× | 1.2× | 6.15 |
| 8k | 64 | 3.7× | 1.2× | 7.99 |
| 16k | 128 | 6.8× | 1.2× | 20.77 |
| 32k | 256 | 11.4× | 1.2× | 76.32 |

**Layer 19** (decay=0.904): Its EMA converges by ~10 chunks, so all eval lengths see the same DeltaW. Safe.

**Layers 10/14** (decay≈0.997): The EMA window is 250–333 chunks, far exceeding the 16-chunk training length. At 32k eval (256 chunks), DeltaW is 11× the training magnitude. The model was trained with DeltaW at ~10% of the base weight norm; at 32k eval it reaches ~120% — a regime the model never learned to handle.

**At 4k**, the extrapolation is only 2×, which the model tolerates. The TTT mechanism provides genuine benefit by retrieving relevant information from the additional context via the fast-weight outer product: `DeltaW @ z_n = Σ vhat_t · <z_t, z_n>` (similarity-weighted retrieval of past target directions).

**At 16k+**, the DeltaW component overwhelms the base weight, producing outputs that deviate significantly from the training distribution.

### Why Not Just Train with Longer Sequences?

This is the correct fix. The spec calls for seq_len=32768 in Phase 3. With 256 chunks during training, the model would learn decay factors and inner learning rates calibrated for the full context range. The extrapolation gap disappears entirely.

---

## 4. Blockers and Open Questions

### 4.1 Fixable with More Compute (Phase 2/3)

- **Sequence length mismatch**: Training on 2048 tokens while evaluating at 32768 creates a 16× chunk-count extrapolation. Training at seq_len=32768 directly addresses this.
- **SWA Transformer degradation**: Also caused by short training sequences. Will resolve at seq_len=32768.
- **More evaluation documents**: Only 5 docs with ≥32k tokens in our validation set. Phase 3 should use more.

### 4.2 Potential Fundamental Concerns

1. **Decay factor drift toward 1.0**: During training, the model pushed layers 10/14 decay factors to 0.996–0.997, effectively trying to memorize all past chunks. Even at seq_len=32768, the model might again learn decay→1.0, creating extrapolation issues if eval sequences exceed training length. A possible mitigation: **cap the maximum decay factor** (e.g., 0.99) to enforce a finite effective memory window. This needs to be tested.

2. **DeltaW scale relative to W0**: The spec's update rule `DeltaW += G` has no inherent mechanism to keep DeltaW small relative to W0. The inner learning rate and decay factor help, but the model still learns to make DeltaW a significant fraction of W0 (~10% at training, up to 120% at long eval). A **normalization** step (e.g., `DeltaW / max(1, ||DeltaW|| / ||W0||)`) could help but changes the method.

3. **Whether TTT can beat vanilla SSM at 32k**: The vanilla SSM already achieves flat ~7.0 PPL at all lengths. Its recurrent state naturally forgets old information (a form of built-in decay). The TTT mechanism adds explicit long-range retrieval, but it remains to be seen whether this provides measurable benefit over the SSM's implicit long-range memory when both are trained at the target sequence length.

4. **Per-layer inner_lr and decay are not in the original spec**: The spec describes a simple `DeltaW += (1/C) * Vhat^T @ Z` update without learnable inner learning rate or decay. These additions were necessary to prevent catastrophic DeltaW explosion, but they represent a modification to the proposed method. The spec's optional clipping (`clip_tau`) alone is insufficient because it bounds per-chunk updates but not cumulative growth.

### 4.3 Next Steps for Phase 2/3

1. Train at seq_len=32768 (the spec's target)
2. Scale to 500M parameters
3. Increase token budget to 5–20B
4. Consider capping decay factor at 0.99 to prevent extrapolation issues
5. Use partition=ailab with H200 GPUs for the full run

---

## 5. Repository Structure

```
ssm_ttt/
├── configs/                     # YAML training configs
│   ├── tiny_pilot_ssm_ttt.yaml
│   ├── tiny_pilot_transformer_swa.yaml
│   └── tiny_pilot_vanilla_ssm.yaml
├── data/
│   ├── __init__.py
│   ├── dataloader.py            # Pre-tokenized dataset + dataloader
│   └── prepare_data.py          # Pile tokenization script
├── models/
│   ├── __init__.py
│   ├── ssm_ttt_model.py         # SSM backbone + TTT layer selection
│   ├── swa_transformer.py       # SWA Transformer baseline
│   ├── target_builder.py        # LM-aligned target (mix_coeffs + W_tgt)
│   └── ttt_wrapper.py           # TTT wrapper: inner_lr, decay, DeltaW loop
├── scripts/
│   ├── build_deps.sh            # Dependency build scripts
│   ├── build_deps_v2.sh
│   ├── count_params.py
│   ├── run_eval_phase1.sh       # SLURM eval script
│   ├── run_phase0_tests.sh
│   ├── run_phase1_all.sh        # SLURM training script
│   └── run_tiny_pilot.sh
├── tests/
│   ├── __init__.py
│   └── test_phase0.py           # Unit tests
├── train.py                     # Main training loop
├── evaluate.py                  # Figure-2 sliding-window PPL evaluation
├── ssm_in_place_ttt_project_spec.md  # Original project specification
└── REPORT.md                    # This file
```
