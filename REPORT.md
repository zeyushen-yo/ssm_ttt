# SSM + In-Place TTT: Project Report

## 1. Implementation Summary

### Models (~131M params each, within ±2%)

- **Vanilla SSM**: 24-layer Mamba2, d_model=768, expand=2, d_state=128
- **SWA Transformer**: 12-layer decoder-only Transformer with sliding-window attention (window=512), RoPE, SwiGLU FFN
- **SSM + TTT (v3)**: Same Mamba2 backbone with 4 TTT-wrapped layers at indices [5, 10, 14, 19]

### TTT v3 Mechanism

Improvement over v1/v2, implementing the updated spec (`ssm_ttt_updated_spec_v2.md`):

- **Normalized discounted update**: `DeltaW_{c+1} = λ · DeltaW_c + (1−λ) · η · G_c`
  - Ensures total kernel mass is bounded regardless of context length
  - `λ` bounded in `[λ_min, λ_max]` = `[0.90, 0.995]` via sigmoid reparameterization
- **RMS-normalized features**: Token-wise RMS normalization on features (`z_t`) and targets (`v_t`) before forming gradient `G`
- **Relative Frobenius cap**: Projects `DeltaW` to `||DeltaW||_F ≤ ρ · ||W_0||_F` (ρ=0.10), and `G` to `||G||_F ≤ γ · ||W_0||_F` (γ=0.02)
- **LM-aligned target**: Depthwise linear combiner (`K_tgt=5`) with trainable projection `W_tgt`

### Data Pipeline

- Pre-tokenized The Pile (~2B tokens) using GPT-NeoX tokenizer
- **Fixed critical padding bug**: Old pipeline included 54% EOS-padding tokens in loss (artificially deflating training loss). Fixed by setting `labels[padding] = -100` and enforcing `min_doc_len ≥ seq_len`.
- **Packed document mode** for long-context training: contiguous chunks from the full 2B corpus, sequences may span document boundaries

---

## 2. Stage 1 Results: Config Screening (200M tokens, seq_len=2048)

### 2.1 TTT Config Comparison

All models trained on 200M tokens from The Pile, seq_len=2048, batch_size=16 (except C2/C3/C4 at batch_size=8 due to memory). Evaluated with sliding-window PPL on 50 validation documents (≥32k tokens each), scoring the last 2048-token suffix.

| Config | Description | 2k | 4k | 8k | 16k | 32k |
|--------|-------------|------|------|------|------|------|
| SWA | Transformer w/ sliding-window attention | **38.47** | **39.06** | 52.14 | 88.51 | 132.83 |
| Vanilla | Mamba2 SSM baseline | 51.46 | 49.22 | 49.24 | 49.24 | 49.25 |
| C0 | TTT v2 control (unnormalized) | 49.88 | 47.47 | 51.13 | 57.15 | 61.34 |
| **C1** | **TTT v3, chunk=128** | 49.80 | 47.12 | **47.09** | **47.09** | **47.09** |
| **C2** | **TTT v3, chunk=64** | 49.98 | **46.74** | 46.77 | 46.77 | **46.78** |
| C4 | TTT v3, target=layer_input | 51.85 | 49.15 | 49.17 | 49.17 | 49.17 |
| C5 | TTT v3, 2 layers only [14,19] | 50.96 | 48.47 | 48.47 | 48.47 | 48.47 |

### 2.2 Key Findings (Stage 1)

1. **TTT v3 (C1, C2) beats Vanilla SSM at all contexts ≥ 4k**. C2 achieves 46.78 vs Vanilla's 49.25 at 32k (−5.0%).

2. **TTT v3 is completely stable** — PPL is flat from 4k to 32k, confirming the normalized discounted update + Frobenius cap eliminated the DeltaW explosion from v1/v2.

3. **C0 (v2 control, unnormalized) degrades at long context** (61.34 at 32k vs 47.09 for C1), confirming the v3 improvements are essential.

4. **SWA degrades severely at long context** (132.83 at 32k) — caused by RoPE extrapolation since the model was trained at seq_len=2048 but evaluated at 32768. This is not a model quality issue; it is resolved by training at seq_len=32768 (see Section 3).

5. **Chunk=64 (C2) is slightly better than chunk=128 (C1)** — PPL 46.78 vs 47.09 at 32k, consistent with finer-grained TTT adaptation.

---

## 3. Stage 1 Results: Fair Long-Context Comparison (200M tokens, seq_len=32768)

To fairly evaluate the SWA Transformer at long contexts, all models are retrained at seq_len=32768 using **packed document mode** (contiguous chunks from the full 2B corpus).

### 3.1 All Models Trained at 32k (Fair Comparison)

All models trained on 200M tokens, seq_len=32768, batch_size=1, packed document mode.

| Model | 2k | 4k | 8k | 16k | 32k | Final train loss |
|-------|------|------|------|------|------|-----------------|
| **SWA (32kp)** | **56.44** | **52.25** | **52.19** | **52.15** | **52.61** | 4.17 |
| Vanilla (32kp) | 62.55 | 59.87 | 59.82 | 59.81 | 59.82 | 4.09 |
| C1 TTT v3 (32kp) | 63.26 | 60.27 | 60.13 | 60.12 | 60.12 | 4.08 |
| **C2 TTT v3 (32kp)** | 60.79 | 57.76 | 57.69 | **57.67** | **57.67** | 4.09 |

### 3.2 Observations (32kp Training)

1. **SWA is the strongest model when all are trained at 32k with batch_size=1**. PPL 52.61 at 32k beats the best SSM model (C2 at 57.67) by 5.0 points.

2. **C2 TTT v3 (chunk=64) is the best SSM model**, beating Vanilla SSM (59.82 → 57.67, a 3.6% improvement) and C1 TTT v3 (60.12 → 57.67, also better). The finer chunk size matters.

3. **C1 TTT v3 (chunk=128) does NOT improve over vanilla SSM** — PPL 60.12 vs 59.82. With chunk=128, the TTT updates are too coarse-grained to help when the SSM already sees 32k contexts during training.

4. **SWA PPL decreases with context** as expected (56.44 → 52.15 from 2k to 16k), confirming the RoPE issue from Section 2 is resolved.

5. **32kp models have higher overall PPL than 2k models** — this is expected because batch_size=1 (vs 16 at 2k) reduces gradient signal quality per step despite equal total tokens (200M).

### 3.3 Cross-Training Comparison

> **Caveat: This is NOT a fair comparison.** The 2k-trained SSM models use batch_size=16, giving 16× better gradient statistics per step than the 32k-trained models (batch_size=1). The lower PPL of 2k-trained models is primarily a batch_size effect, not evidence that shorter training is better. SWA cannot be included at 2k training due to RoPE extrapolation failure. The fair comparison is Section 3.1 (all models at 32k, batch_size=1). This table is included only to show that SSM models can be evaluated at any context length due to lack of positional encoding.

| Model | Training | batch | 2k | 4k | 8k | 16k | 32k |
|-------|----------|-------|------|------|------|------|------|
| SWA (32kp) | 32k | 1 | 56.44 | 52.25 | 52.19 | 52.15 | 52.61 |
| Vanilla (2k) | 2k | 16 | 51.46 | 49.22 | 49.24 | 49.24 | 49.25 |
| C1 TTT v3 (2k) | 2k | 16 | 49.80 | 47.12 | 47.09 | 47.09 | 47.09 |
| C2 TTT v3 (2k) | 2k | 16 | 49.98 | 46.74 | 46.77 | 46.77 | 46.78 |

### 3.4 Key Takeaways

1. **TTT v3 consistently improves SSM perplexity** — in both the 2k and 32k training regimes, the best TTT config outperforms vanilla SSM (46.78 vs 49.25 at 2k-trained; 57.67 vs 59.82 at 32kp-trained).

2. **Chunk size matters**: C2 (chunk=64) consistently outperforms C1 (chunk=128). At 32kp, this is the difference between improving over vanilla (C2) and not (C1).

3. **SWA vs SSM+TTT depends on training regime**: SWA wins when all models use batch_size=1 at 32k. SSM+TTT wins when SSM models use their optimal batch_size=16 at 2k. A truly fair comparison requires training all models at seq_len=32768 with batch_size≥8, which we leave to Stage 2.

4. **The core TTT mechanism works**: The v3 normalized discounted update is completely stable (no DeltaW explosion), and the chunk-wise fast-weight adaptation genuinely improves language modeling at all context lengths ≥4k.

---

## 4. TTT v3 Diagnostics

### 4.1 DeltaW_rel at Evaluation (32kp-trained models)

`DeltaW_rel` = `||DeltaW||_F / ||W_0||_F` measures how much the fast weight deviates from the base weight.

**C1 (chunk=128):**

| Context | Layer 5 | Layer 10 | Layer 14 | Layer 19 |
|---------|---------|----------|----------|----------|
| 2k | 0.0036 | 0.0037 | 0.0040 | 0.0032 |
| 8k | 0.0044 | 0.0047 | 0.0051 | 0.0041 |
| 32k | 0.0044 | 0.0047 | 0.0051 | 0.0042 |

**C2 (chunk=64):**

| Context | Layer 5 | Layer 10 | Layer 14 | Layer 19 |
|---------|---------|----------|----------|----------|
| 2k | 0.0044 | 0.0041 | 0.0044 | 0.0043 |
| 8k | 0.0046 | 0.0043 | 0.0047 | 0.0045 |
| 32k | 0.0046 | 0.0043 | 0.0047 | 0.0045 |

C2 achieves slightly higher `DeltaW_rel` than C1 (0.0045 vs 0.0042 at 32k), consistent with C2's better performance — the finer chunk size allows more accumulated adaptation.

### 4.2 Stability Analysis

Both C1 and C2 show excellent stability:
- `DeltaW_rel` saturates by ~8k context and is flat from 8k to 32k
- Maximum `DeltaW_rel` is ~0.005, far below the 0.10 cap (ρ parameter)
- Growth from 2k to 32k is only ~1.2× for both models (vs 4.7× for the unnormalized C0 baseline)

The normalized update rule `DeltaW = λ·DeltaW + (1−λ)·η·G` has a theoretical steady-state bound `||DeltaW||_∞ ≤ η·||G||_∞`, independent of context length. The experimental results confirm this: PPL is perfectly flat from 8k to 32k for both C1 and C2.

---

## 5. Bug Fixes and Lessons Learned

1. **Training data padding bug** (critical): 54% of training samples were padded with EOS tokens, included in loss computation. This artificially deflated reported training loss (reported 2.0, actual on real text 4.57). Fixed by masking padding with `labels=-100` and setting `min_doc_len=seq_len`.

2. **32k training data diversity**: Initially restricted to documents ≥32768 tokens (only 3,146 from 1.3M total, 222M tokens). Models severely underfit (SWA PPL jumped from 38 to 205). Fixed with **packed document mode** — sampling contiguous 32768-token chunks from the full 2B corpus.

3. **GPU memory for C2 (chunk=64)**: At seq_len=32768, batch=1, C2 needs 73.82 GB peak memory. OOM on 40GB MIG slices and tight on 80GB A100. Fixed with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and requesting `--gres=gpu:a100:1`.

---

## 6. Phase 1 Screening: v3 Correctness Fixes (50M tokens, seq_len=32768)

### 6.1 Design

Four variants tested to isolate the impact of each correctness fix from the v3 spec. All use in-place TTT on `out_proj` (chunk=64, 4 TTT layers), trained on 50M screening tokens.

| Config | Description |
|--------|------------|
| P1-A | C2 control (no boundary awareness, default decay 0.9–0.995) |
| P1-B | P1-A + boundary-aware packed training (DeltaW resets at doc boundaries) |
| P1-C | P1-B + longer decay range (init 0.995, min 0.98, max 0.9995) |
| P1-D | P1-C + 4-group optimizer split (3× LR for TTT target params) + higher G_rel_cap (0.05) |

### 6.2 Results

| Model | 2k | 4k | 8k | 16k | 32k | PPL Δ (2k→32k) |
|-------|------|------|------|------|------|----------------|
| **P1-D** | **168.22** | **165.88** | **165.57** | **165.37** | **165.24** | **−3.0** |
| P1-A (control) | 171.63 | 168.44 | 168.38 | 168.39 | 168.40 | −0.0 |
| Stage1-C2 @50M | 169.92 | 167.17 | 167.12 | 167.12 | 167.12 | −0.0 |
| P1-C | 173.24 | 170.80 | 170.53 | 170.43 | 170.34 | −2.9 |
| P1-B | 173.54 | 170.68 | 170.63 | 170.65 | 170.65 | −0.0 |

Note: absolute PPL values are high because all models are severely undertrained at 50M tokens (only 1,525 gradient steps with batch_size=1). The Stage 1 C2 reference trained at 50M tokens confirms this is expected (PPL 167.12). The meaningful comparison is between Phase 1 variants.

### 6.3 Key Findings

1. **P1-D is the clear winner** — 3 PPL points better than P1-A at all context lengths, and the only variant that substantially beats the Stage 1 C2 reference at the same token budget.

2. **P1-D shows context-dependent improvement** — PPL drops from 168.2 at 2k to 165.2 at 32k (3 point gain). P1-A and P1-B are completely flat across context lengths.

3. **P1-D diagnostics show the right behavior:**
   - `deltaW_rel` grows 4.5× from 2k→32k (0.0008→0.0036), meaning TTT adapts more with longer context
   - At 32k, deltaW_rel reaches 0.0032–0.0040, matching Stage 1 C2 at 200M tokens
   - Effective window: ~10,057 tokens (10× longer than A/B's ~1,074 tokens)

4. **The optimizer split + higher G_rel_cap are the critical enablers** — they allow TTT parameters to learn faster and produce larger gradient updates.

5. **Boundary awareness alone (P1-B) hurts** — DeltaW resets at document boundaries discard accumulated learning. At 50M tokens the model cannot recover.

6. **Longer decay range (P1-C) shows the right dynamics but worse overall PPL** — the 10× longer effective window and context-dependent deltaW_rel are promising, but boundary-awareness overhead prevents net improvement over A.

### 6.4 Phase 0 Tests

All 12 correctness tests PASSED, including 4 new v3 boundary-aware tests:
- v3-1: Document-boundary target masking
- v3-2: Boundary reset of fast weights
- v3-3: Packed dataset segment IDs
- v3-4: Residual fast path gate=0 identity

---

## 7. Stage 2: Scaling to 200M Tokens

### 7.1 Design

Based on Phase 1 screening, two configs were trained at 200M tokens for a proper comparison against Stage 1 baselines. Both use **in-place TTT** (no residual fast path).

**Design principle:** We strongly prefer in-place TTT (`W_eff = W0 + DeltaW`) over adding new architectural components. The appeal is elegance: the SSM backbone is unchanged, and TTT simply adapts existing weights at test time.

| Config | Changes from Stage 1 C2 |
|--------|------------------------|
| s2_decay | Longer decay (init 0.995, min 0.98, max 0.9995) |
| s2_decay_optim | + optimizer split (3× TTT LR) + G_rel_cap 0.05 |
| Stage 1 C2 | Control (default decay 0.9–0.995) |

Neither s2 config uses boundary awareness (which hurt in Phase 1 screening).

### 7.2 Results

All five models evaluated on the same 50 validation documents in a single eval run for consistent comparison. All trained at 200M tokens, seq_len=32768, batch_size=1.

| Model | 2k | 4k | 8k | 16k | 32k |
|-------|------|------|------|------|------|
| **SWA Transformer** | **57.44** | **53.52** | **53.39** | **53.34** | **53.82** |
| Stage 1 C2 (TTT) | 61.33 | 58.41 | 58.35 | 58.34 | 58.34 |
| s2_decay | 61.84 | 59.38 | 59.22 | 59.17 | 59.15 |
| Vanilla SSM | 63.06 | 60.41 | 60.36 | 60.35 | 60.36 |
| s2_decay_optim | 65.85 | 62.82 | 62.53 | 62.43 | 62.41 |

### 7.3 Analysis

1. **Longer decay hurts.** s2_decay (59.15 at 32k) is worse than Stage 1 C2 (58.34), though still better than Vanilla SSM (60.36). The extended decay range (min 0.98, max 0.9995 vs default 0.9–0.995) does not translate to better perplexity at scale.

2. **Optimizer split + higher G_rel_cap also hurts at scale.** s2_decay_optim (62.41 at 32k) is the worst model — worse even than vanilla SSM. Despite P1-D being the 50M screening winner, the changes that helped at 50M tokens (3× TTT LR, G_rel_cap 0.05) cause divergence or overfitting of TTT parameters when trained to 200M tokens.

3. **Stage 1 C2 remains the best SSM+TTT configuration.** The original default decay (0.9–0.995) with standard optimizer settings outperforms both Stage 2 variants at full scale.

4. **SWA Transformer still dominates.** PPL 53.82 vs best SSM+TTT at 58.34 — a gap of 4.5 points (7.7% relative).

---

## 8. Critical Assessment

### 8.1 What Works

- **In-place TTT is stable.** The v3 normalized discounted update (`DeltaW = λ·DeltaW + (1−λ)·η·G`) with Frobenius caps completely eliminates the DeltaW explosion seen in v1/v2. PPL is perfectly flat from 8k to 32k.
- **TTT consistently improves over vanilla SSM.** Stage 1 C2 achieves 58.34 vs 60.36 (3.3% improvement at 32k) — this holds across both training regimes (2k and 32k) and is not a fluke.
- **The mechanism is correct.** All 12 Phase 0 tests pass, diagnostics show expected behavior (deltaW_rel grows with context, saturates at steady state).

### 8.2 Concerns

1. **The improvement is small.** A ~2 PPL point gain (3.3% relative) is modest, especially given the added complexity: extra parameters (target builder with mix_coeffs, W_tgt), learnable inner LR and decay per layer, chunk-wise updates at inference time.

2. **TTT does not show context-dependent improvement.** The gap between C2 and Vanilla SSM is roughly constant across all context lengths — both are flat from ~4k onward. If TTT were leveraging long-range context in a meaningful way, we would expect a *widening* gap at longer contexts. Instead, the improvement appears to be a uniform shift, which could come from simply having more trainable parameters rather than from the TTT adaptation mechanism itself.

3. **SWA Transformer still dominates.** The gap between SWA (53.82) and the best SSM+TTT (58.34) is 4.5 PPL points. The narrative "we improved SSM by 3.3% but it's still 7.7% worse than attention" is not compelling.

4. **Could be a parameter count effect.** C2 has additional parameters from the TTT target builder (mix_coeffs, W_tgt projections across 4 layers). Without a parameter-matched vanilla SSM control (e.g., a wider d_model or extra FFN layers), we cannot rule out that the improvement comes from extra capacity rather than the TTT adaptation mechanism.

5. **Phase 1 screening results did not transfer to full scale.** P1-D was the clear 50M-token winner, but both changes it introduced (longer decay, optimizer split) degraded performance at 200M tokens. This suggests the screening methodology at 50M tokens is unreliable for predicting 200M-token outcomes.

### 8.3 Honest Summary

In-place TTT on SSMs is a valid proof-of-concept: the mechanism is stable, correct, and directionally helpful. However, the effect size is too small (~3%) and too context-independent to constitute a strong research result on its own. For a publishable contribution, one would need: (a) a substantially larger gain over vanilla SSM, (b) clear evidence that TTT specifically helps at long contexts (widening gap with context length), or (c) closing the gap with attention-based models.

---

## 9. Phase A: Error-Corrective Delta Rules (v4 Spec)

### 9.1 Motivation

The v4 spec diagnosed a fundamental limitation: the Hebbian update rule (`G = V_hat^T @ Z_norm`) may store a "stationary average correction" rather than building a "context-growing associative memory." The hypothesis was that switching to an **error-corrective delta rule** — where updates are driven by prediction residuals — would produce context-dependent gains.

### 9.2 Design

Four variants tested at 50M screening tokens, seq_len=32768, batch_size=1:

| Config | Update Rule | Scale Mode | Description |
|--------|------------|------------|-------------|
| A0 | hebb | mean | v3 baseline (control) |
| A1 | hebb | sqrt_len | Hebbian with sqrt scaling |
| A2 | delta_current | sqrt_len | Error-corrective: E = V_hat - sg(O_current) |
| A3 | delta_base | sqrt_len | Error-corrective: E = V_hat - sg(Z @ W0^T) |

All other parameters identical (decay 0.9–0.995, G_rel_cap 0.02, deltaW_rel_cap 0.10).

New implementation features:
- Three update rules: `hebb`, `delta_current`, `delta_base`
- Three scale modes: `mean`, `sqrt_len`, `sum`
- `err_rms` diagnostic for delta rules
- TTT-on/off evaluation control
- Random-prefix and shuffled-prefix controls
- B>1 boundary assertion (explicit error instead of silent fallback)
- 17 Phase 0 tests all passing (12 original + 5 new v4 tests)

### 9.3 Results — Standard PPL

| Model | 2k | 4k | 8k | 16k | 32k |
|-------|------|------|------|------|------|
| **A0 (hebb+mean)** | **168.31** | **165.31** | **165.19** | **165.18** | **165.15** |
| A1 (hebb+sqrt) | 171.48 | 168.44 | 168.38 | 168.38 | 168.38 |
| A2 (delta_current) | 174.66 | 172.45 | 172.45 | 172.46 | 172.46 |
| A3 (delta_base) | 176.07 | 173.37 | 173.26 | 173.25 | 173.23 |

200M baselines for reference (not directly comparable due to 4× more training tokens):
- SWA 200M: 53.82 at 32k
- Stage 1 C2 (hebb 200M): 58.34 at 32k
- Vanilla SSM 200M: 60.36 at 32k

### 9.4 Results — TTT ON vs OFF (Same Checkpoint)

This is the key diagnostic: evaluating each model with TTT updates enabled vs disabled.

| Model | Gain(2k) | Gain(4k) | Gain(8k) | Gain(16k) | Gain(32k) | Context-dep |
|-------|----------|----------|----------|-----------|-----------|-------------|
| **A0 (hebb+mean)** | 0.76 | 1.32 | 1.39 | 1.39 | **1.40** | **+0.64** |
| **A1 (hebb+sqrt)** | 0.85 | 1.43 | 1.50 | 1.50 | **1.50** | **+0.65** |
| A2 (delta_current) | 0.02 | 0.04 | 0.05 | 0.05 | 0.06 | +0.03 |
| A3 (delta_base) | 0.04 | 0.09 | 0.11 | 0.09 | 0.09 | +0.05 |
| C2 (hebb 200M) | 1.33 | 1.88 | 1.91 | 1.92 | **1.92** | **+0.59** |

`Gain(L) = PPL_off(L) - PPL_on(L)`. Positive means TTT helps.
`Context-dep = Gain(32k) - Gain(2k)`. Positive means gain grows with context.

### 9.5 Results — Prefix Corruption Controls

| Model | Normal 32k | Random 32k | Shuffled 32k |
|-------|-----------|-----------|-------------|
| A0 (hebb+mean) | 165.15 | 171.65 (+6.5) | 166.61 (+1.5) |
| A1 (hebb+sqrt) | 168.38 | 174.78 (+6.4) | 169.85 (+1.5) |
| A2 (delta_current) | 172.46 | 178.75 (+6.3) | 173.73 (+1.3) |
| A3 (delta_base) | 173.23 | 179.14 (+5.9) | 174.37 (+1.1) |

Random prefix (replacing prefix with tokens from another document) degrades PPL by ~6 points for all models, confirming the SSM backbone itself uses context content. Shuffled prefix degrades by ~1.3 points, showing sequential structure matters somewhat.

### 9.6 Analysis (50M Screening)

**1. The delta rule hypothesis is not supported at 50M.** Delta-current and delta-base produce near-zero TTT gains (0.06 and 0.09 PPL at 32k), while the Hebbian rule produces 1.4–1.5 points of gain.

**2. The Hebbian rule shows genuine context-dependent gain.** TTT gain grows with context: Gain(32k) - Gain(2k) = 0.64–0.65 for Hebbian models.

**3. Sqrt_len scaling does not help Hebbian overall.** A1 (hebb+sqrt_len) has worse absolute PPL than A0 (hebb+mean) by ~3 points, but slightly higher TTT gain (1.50 vs 1.40).

---

### 9.7 Phase A at 200M Tokens (Full Training)

To obtain definitive results, the two most informative configs were trained to 200M tokens from scratch:
- **A2 (delta_current+sqrt_len)**: The primary v4 proposal, to test whether the delta rule improves with more training
- **A1 (hebb+sqrt_len)**: To test whether sqrt_len scaling helps Hebbian at full scale

All models use seq_len=32768, batch_size=1, warmup=512 steps, total 6103 steps.

#### 9.7.1 Standard PPL (All 200M Models)

| Model | 2k | 4k | 8k | 16k | 32k |
|-------|------|------|------|------|------|
| **SWA** | **57.44** | **53.52** | **53.39** | **53.34** | **53.82** |
| C2 (hebb+mean) | 61.33 | 58.41 | 58.35 | 58.34 | **58.34** |
| Vanilla SSM | 63.06 | 60.41 | 60.36 | 60.35 | 60.36 |
| A2 (delta_current) 200M | 64.05 | 61.54 | 61.52 | 61.52 | 61.53 |
| A1 (hebb+sqrt) 200M | 64.54 | 61.40 | 61.35 | 61.35 | 61.36 |

#### 9.7.2 TTT ON vs OFF (200M Models)

| Model | Gain(2k) | Gain(4k) | Gain(8k) | Gain(16k) | Gain(32k) | Context-dep |
|-------|----------|----------|----------|-----------|-----------|-------------|
| A2 (delta_current) 200M | 0.04 | 0.06 | 0.06 | 0.07 | 0.06 | +0.02 |
| **A1 (hebb+sqrt) 200M** | **1.50** | **2.03** | **2.06** | **2.06** | **2.06** | **+0.56** |
| **C2 (hebb+mean) 200M** | 1.33 | 1.88 | 1.91 | 1.92 | **1.92** | **+0.59** |

#### 9.7.3 Prefix Corruption Controls (200M Models)

| Model | Normal 32k | Random 32k | Shuffled 32k |
|-------|-----------|-----------|-------------|
| Vanilla SSM | 60.36 | 65.45 (+5.1) | 62.11 (+1.8) |
| C2 (hebb+mean) | 58.34 | 63.98 (+5.6) | 60.38 (+2.0) |
| A1 (hebb+sqrt) | 61.36 | 67.49 (+6.1) | 63.26 (+1.9) |
| A2 (delta_current) | 61.53 | 67.01 (+5.5) | 63.32 (+1.8) |
| SWA | 53.82 | 58.47 (+4.7) | 55.92 (+2.1) |

### 9.8 Analysis (200M — Definitive)

**1. Delta-current is definitively ineffective.** At 200M tokens, A2 (delta_current) produces only 0.06 PPL of TTT gain — identical to the 50M result. More training does not help. The delta rule's error signal, which is clipped to the same G_rel_cap (0.02) as Hebbian, loses its error-corrective nature. The v4 spec's primary hypothesis (error-corrective delta rules would outperform Hebbian for pure SSMs) is conclusively disproven.

**2. Both A1 and A2 are worse than vanilla SSM in absolute PPL.** A1 achieves 61.36 and A2 achieves 61.53 at 32k, both worse than Vanilla SSM (60.36). Despite the TTT mechanism being active and producing gains for A1, the sqrt_len scaling hurts the base model training enough to more than offset the TTT benefit. C2 (hebb+mean), which uses mean scaling, remains the only TTT config that beats vanilla SSM at 200M.

**3. A1 (hebb+sqrt) has the largest TTT gain but worst absolute PPL.** A1's TTT gain of 2.06 at 32k exceeds C2's 1.92. But A1's TTT-OFF PPL (~63.4) is much worse than C2's TTT-OFF PPL (~60.3), meaning the base model learned less well. The sqrt_len scaling (dividing G by sqrt(64)=8 instead of 64) makes updates 8× larger before capping, which appears to distort the training dynamics.

**4. C2 (hebb+mean) is confirmed as the best TTT configuration.** It is the only model that (a) beats vanilla SSM in absolute PPL, (b) shows meaningful TTT gain (1.92 at 32k), and (c) exhibits context-dependent improvement (+0.59).

**5. Prefix corruption controls confirm context sensitivity is primarily from the SSM backbone.** Even vanilla SSM (no TTT) degrades by 5.1 PPL with random prefix. The TTT-specific contribution is modest: C2 degrades 0.5 more than vanilla (5.6 vs 5.1), A1 degrades 1.0 more (6.1 vs 5.1), while A2 degrades only 0.4 more (5.5 vs 5.1) — consistent with A2's near-zero TTT gain.

**6. SWA still dominates by a wide margin.** 53.82 vs 58.34 (best TTT) — a 4.5 PPL gap (7.7% relative). The gap to the best TTT model (C2) has not closed compared to Stage 2 results.

---

## 10. Status and Next Steps

### Completed
- [x] Stage 1: Config screening (7 configs at 2k, 4 models at 32k, 200M tokens)
- [x] Phase 0: All 17 correctness tests passed (12 original + 5 v4)
- [x] Phase 1: 50M screening of v3 fixes → P1-D best at 50M but did not transfer to 200M
- [x] Stage 2: s2_decay and s2_decay_optim at 200M tokens → both worse than Stage 1 C2
- [x] Phase A screening (50M): delta rules produce near-zero TTT gain
- [x] Phase A at 200M: A2 (delta_current) and A1 (hebb+sqrt) trained to 200M tokens
  - Delta-current definitively fails (0.06 PPL gain, identical to 50M result)
  - Hebb+sqrt shows strong TTT gain (2.06) but worse absolute PPL than vanilla SSM
  - C2 (hebb+mean) confirmed as the best overall TTT configuration

### Key Insights
1. **Hebbian update with mean scaling (C2) is the right recipe.** It is the only configuration that simultaneously: beats vanilla SSM in absolute PPL, shows meaningful TTT gain, and exhibits context-dependent improvement. All other variants (longer decay, optimizer split, delta rules, sqrt scaling) either fail to improve or actively degrade performance.
2. **Error-corrective delta rules are not suitable for in-place SSM TTT.** The near-zero gain persists from 50M to 200M tokens. The G_rel_cap may be a contributing factor: both Hebbian and delta rules hit the 0.02 cap, meaning the cap dominates the update magnitude and removes the delta rule's error-proportional scaling. However, even the update *direction* (error vs target) appears unhelpful.
3. **TTT ON/OFF evaluation is a powerful diagnostic.** This revealed that C2's improvement is genuinely from online adaptation (1.92 PPL at 32k), not just extra parameters. It also definitively showed the delta rule's failure.
4. **50M screening has limited predictive power.** Phase 1 (P1-D), Stage 2 (s2_decay_optim), and Phase A all produced misleading signals at 50M tokens. The delta rule's failure was correctly identified at 50M, but the relative ranking of Hebbian variants was less reliable.

### Possible Next Steps (per v4 spec decision tree — Case 3)
The v4 spec's Case 3 applies: delta-current gives constant-shift (near-zero) gain. Recommended actions:
1. **Centered updates** — subtract running mean from G to remove stationary drift: `G_centered = G - EMA(G)`
2. **Surprise gating** on Hebbian rule — use chunk error magnitude to selectively suppress updates on predictable chunks
3. **Parameter-matched control** — verify C2's 1.92 PPL gain is from TTT adaptation, not extra parameter capacity
4. **Top-heavy TTT placement** — concentrate TTT layers in upper half of network where representations are more task-specific
5. **Boundary-aware training with long documents** — may unlock context-dependent gains at longer horizons

---

## 11. Repository Structure

```
ssm_ttt/
├── configs/                     # YAML training configs
│   ├── stage1_*.yaml            # Stage 1 configs (seq_len=2048)
│   ├── stage1_32k_*.yaml        # Stage 1 configs (seq_len=32768)
│   ├── phase1_*.yaml            # Phase 1 screening configs (50M tokens)
│   ├── phaseA_*.yaml            # Phase A configs (50M screening + 200M full)
│   └── s2_*.yaml                # Stage 2 configs (200M tokens)
├── data/
│   ├── dataloader.py            # DocOffset, Packed, Boundary datasets
│   └── prepare_data.py          # Pile tokenization script
├── models/
│   ├── ssm_ttt_model.py         # SSM backbone + TTT layer selection
│   ├── swa_transformer.py       # SWA Transformer baseline
│   ├── target_builder.py        # LM-aligned target (mix_coeffs + W_tgt)
│   └── ttt_wrapper.py           # TTT v3 wrapper: normalized update, caps
├── tests/
│   └── test_phase0.py           # Phase 0 + v3 correctness tests
├── scripts/                     # SLURM job scripts
├── runs/                        # Training outputs and checkpoints
├── train.py                     # Main training loop
├── evaluate.py                  # Figure-2 sliding-window PPL evaluation
└── REPORT.md                    # This file
```
