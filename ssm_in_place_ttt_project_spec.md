# SSM In-Place Test-Time Training Project Spec

## Document purpose

This document is a concrete execution spec for implementing and evaluating **in-place test-time training (TTT) on a state-space model (SSM)**. It is written so that an engineering/research agent can follow it without needing additional interpretation.

The immediate target is a **Figure-2-style result**: show that a matched-parameter **SSM + in-place TTT** achieves **lower sliding-window perplexity than a matched sliding-window-attention (SWA) Transformer baseline** at long context. A vanilla SSM baseline must also be included.

This plan is intentionally conservative. The goal is to maximize the probability of getting a clean signal fast, not to explore the whole design space at once.

---

## 1. Project background

The appended paper introduces **In-Place Test-Time Training** for Transformers by:

1. repurposing the **final projection matrix** of selected MLP blocks as the fast weights,
2. updating those fast weights **chunkwise** rather than tokenwise,
3. using an **LM-aligned target** instead of a generic reconstruction target, and
4. showing strong long-context gains, including a Figure-2-style **sliding-window perplexity vs context length** evaluation.

For this SSM project, the direct analogue is:

- do **not** adapt the recurrent/state-update parameters of the SSM in the first pass,
- do adapt the **final linear output projection inside selected SSM blocks**,
- do use an **NTP-aligned chunk update**,
- do train and evaluate with the same protocol across models.

### Why this is the right first attempt

A pure SSM block usually still ends in a linear projection from an internal feature space back to the model dimension. That projection is the closest architectural match to the paper’s `W_down` choice. Adapting that projection preserves the frozen sequence-processing dynamics of the SSM while still giving a high-capacity fast associative memory.

The first pass must therefore treat the SSM block as:

- a **frozen feature extractor up to pre-output features**,
- plus a **fast linear readout** that can be updated online.

Do **not** adapt the SSM transition/state parameters in the first pass.

---

## 2. Primary hypothesis

### Main hypothesis

Let the SSM produce pre-output features `z_t^l` at selected layers `l`. If we adapt the block output projection in-place with a chunkwise outer-product update aligned to **next-token prediction**, then the model will accumulate a useful fast-weight memory over prior features. This should improve long-context utilization and lower sliding-window perplexity relative to:

1. a matched **SWA Transformer baseline**, and
2. the same SSM backbone **without** TTT.

### Required first success criterion

The first deliverable is considered successful only if all of the following are true:

1. The plot contains **at least three curves**: `SWA Transformer`, `Vanilla SSM`, and `SSM + In-Place TTT`.
2. The models are trained with the **same tokenizer, data, optimizer family, sequence length, and token budget**.
3. Parameter counts are matched within **±5%**.
4. The `SSM + In-Place TTT` curve is **below** the `SWA Transformer` curve at **at least two** of the long-context points from `{8k, 16k, 32k}`.
5. The `SSM + In-Place TTT` curve is also below the `Vanilla SSM` curve at those same points.

If these conditions are not met, the result is not a successful first deliverable.

---

## 3. Non-negotiable implementation decisions

These are the default decisions. Do not change them in the first pass.

1. **Backbone family**: use a Mamba-like decoder-only SSM with an explicit block output projection (`out_proj`).
2. **Fast weight location**: adapt only the selected block’s `out_proj`.
3. **Do not adapt** the SSM recurrence/state-update parameters in the first pass.
4. **Target source for the first figure**: use **token embeddings**, not hidden states. This matches the from-scratch track most closely.
5. **Update granularity**: chunkwise, not tokenwise.
6. **Reference implementation**: sequential chunk loop first. Do not start with prefix-scan/context-parallel optimization.
7. **Document resets**: reset fast weights at every document boundary.
8. **No document packing** unless a correct reset-mask implementation exists.
9. **Outer training objective**: standard next-token cross-entropy for the full model.
10. **Evaluation metric**: Figure-2-style sliding-window perplexity.

---

## 4. Exact mathematical specification

This section defines the exact math to implement.

### 4.1 Notation

- `B`: batch size
- `T`: sequence length
- `C`: chunk size
- `L`: number of model layers/blocks
- `d`: model dimension
- `m_l`: pre-output feature dimension of SSM block `l`
- `x_t`: token id at position `t`
- `e_t = E[x_t] in R^d`: token embedding at position `t`
- `H^l in R^(B x T x d)`: input residual stream to layer `l`
- `Z^l in R^(B x T x m_l)`: pre-output SSM features of layer `l`, computed by the frozen part of the block
- `W_0^l in R^(d x m_l)`: pretrained/base block output projection for layer `l`
- `DeltaW_{b,c}^l in R^(d x m_l)`: fast-weight state for batch item `b`, layer `l`, before chunk `c`
- `O_{b,c}^l in R^(C x d)`: chunk output after applying the adapted projection
- `Vhat_{b,c}^l in R^(C x d)`: LM-aligned target directions

### 4.2 Which layers get TTT

Let `S` be the set of TTT-enabled layers.

For the main run, use **exactly 4 TTT-enabled layers**, evenly spaced through depth.

If the model has `L` blocks, choose:

- `round(L/5)`
- `round(2L/5)`
- `round(3L/5)`
- `round(4L/5)`

then clamp to valid block indices and remove duplicates if needed.

This keeps the TTT state size comparable across backbones and avoids overcomplicating the first run.

### 4.3 Block factorization

For every TTT-enabled layer `l`, refactor the block conceptually into:

1. a **frozen pre-output map** that produces `z_t^l in R^(m_l)`, and
2. the **final output projection** `W_0^l`.

The standard block output is:

```math
 o_t^l = W_0^l z_t^l
```

The adapted block output is:

```math
 o_t^l = (W_0^l + DeltaW_t^l) z_t^l
```

where `DeltaW_t^l` is piecewise constant within a chunk.

### 4.4 Chunk partition

Partition the sequence into non-overlapping chunks of size `C`:

```math
 [1:T] = c_1 \cup c_2 \cup ... \cup c_K
```

where each chunk contains contiguous positions.

For batch item `b`, layer `l`, and chunk `c`, define:

- `Z_{b,c}^l in R^(C x m_l)`
- `Vhat_{b,c}^l in R^(C x d)`

### 4.5 Target source for the main figure

For the **from-scratch Figure-2 track**, define the source sequence as token embeddings:

```math
 q_t = e_t = E[x_t] in R^d
```

Later, for continual-training or warm-start experiments, replace `q_t` with the **layer input hidden state** `h_t^l`, but **do not do that in the first figure**.

### 4.6 LM-aligned target builder

Use a learned future-looking depthwise linear combiner plus a trainable projection.

For each TTT-enabled layer `l`, choose kernel size `K_tgt = 5` and define learned vectors

```math
 d_1^l, d_2^l, d_3^l, d_4^l, d_5^l in R^d
```

with elementwise multiplication. Then define:

```math
 u_t^l = sum_{j=1}^{5} (d_j^l ⊙ q_{t+j})
```

and

```math
 vhat_t^l = u_t^l W_tgt^l
```

where

```math
 W_tgt^l in R^(d x d)
```

is trainable.

#### Boundary rule

If `t + j` exceeds the **end of the current chunk** or the **end of the current document**, set `q_{t+j} = 0` for that term.

This rule is mandatory. It is how chunkwise execution remains equivalent to a strictly causal sequential process.

#### Exact next-token special case

The exact next-token target is the special case:

```math
 d_1^l = 1, d_2^l = d_3^l = d_4^l = d_5^l = 0, W_tgt^l = I
```

which gives:

```math
 vhat_t^l = e_{t+1}
```

The first full run should use the learned `K_tgt = 5` target. The exact next-token target should be implemented only as a sanity-check ablation.

### 4.7 Apply step

For each batch item `b`, TTT-enabled layer `l`, and chunk `c`:

```math
 O_{b,c}^l = Z_{b,c}^l (W_0^l + DeltaW_{b,c}^l)^T
```

This produces the chunk output in model dimension.

### 4.8 Update objective

Use the LM-aligned similarity objective on each chunk:

```math
 J_{b,c}^l = < O_{b,c}^l, Vhat_{b,c}^l >_F
```

Equivalently, define the loss as:

```math
 L_{b,c}^l = - < O_{b,c}^l, Vhat_{b,c}^l >_F
```

This is not the outer LM loss. It is only the internal fast-weight update objective.

### 4.9 Recommended update rule

Define the chunk update matrix:

```math
 G_{b,c}^l = (1/C) (Vhat_{b,c}^l)^T Z_{b,c}^l
```

Then update fast weights by:

```math
 DeltaW_{b,c+1}^l = DeltaW_{b,c}^l + G_{b,c}^l
```

with

```math
 DeltaW_{b,1}^l = 0
```

at the start of every document.

#### Important note

The paper-equivalent update is the unnormalized form:

```math
 DeltaW_{b,c+1}^l = DeltaW_{b,c}^l + (Vhat_{b,c}^l)^T Z_{b,c}^l
```

Using the **mean** over chunk instead of the **sum** is recommended here because it makes the update scale less sensitive to chunk size. This does **not** change the method qualitatively; it only normalizes the effective learning rate.

### 4.10 Optional clipping rule

Use the following only if updates become unstable:

```math
 G_{b,c}^l <- min(1, tau / ||G_{b,c}^l||_F) * G_{b,c}^l
```

before the fast-weight update.

Default policy:

- for the first from-scratch run: **clip off**,
- if there are NaNs, exploding norms, or severe evaluation instability: turn clip **on** with `tau = 1.0` for the mean-normalized update.

Do not use clipping unless there is evidence it is needed.

### 4.11 Outer language-model training objective

The model is still trained end-to-end with standard next-token cross-entropy:

```math
 L_LM = - sum_t log p_theta(x_{t+1} | x_{<=t})
```

The TTT mechanism lives **inside the forward pass**. The outer objective remains standard LM training.

### 4.12 Retrieval interpretation

After processing earlier chunks, the fast weight in one layer is:

```math
 DeltaW_n^l = sum_{t < n} vhat_t^l (z_t^l)^T
```

So at a later position `n`, the TTT contribution becomes:

```math
 delta o_n^l = DeltaW_n^l z_n^l = sum_{t < n} vhat_t^l < z_t^l, z_n^l >
```

This is the key mechanism: later features retrieve a similarity-weighted combination of earlier target directions.

---

## 5. Exact implementation requirements

### 5.1 Required block interface

For each TTT-enabled SSM block, expose the following tensors:

1. `layer_input` in shape `[B, T, d]`
2. `pre_out_features` in shape `[B, T, m_l]`
3. `base_out_proj_weight` in shape `[d, m_l]`

If the codebase does not naturally expose `pre_out_features`, refactor it until it does.

### 5.2 Per-batch-item fast state

Fast weights must be maintained **per sequence item**, not shared across the batch.

Required shape:

```text
DeltaW[layer][batch] : [d, m_l]
```

Never share a single `DeltaW` across the batch.

### 5.3 Precision rule

Compute the following in **fp32** even if the model runs in bf16:

- `DeltaW`
- `G`
- `W_tgt`
- target-builder parameters `d_j`

Then cast the applied output back to model dtype after the matrix multiply if needed.

### 5.4 Initialization rule

For every TTT-enabled layer:

- initialize all depthwise future-mixing coefficients `d_j^l` to zero,
- initialize `W_tgt^l` as a **diagonal-only** matrix with off-diagonals zero and diagonal entries drawn from the model’s standard linear initializer range,
- copy `W_0^l` from the base block output projection,
- initialize `DeltaW = 0` at the start of each sequence/document.

This guarantees the TTT path is initially a near-no-op.

### 5.5 No document packing in phase 1

During the first phase:

- each training sample must correspond to **one contiguous document segment**,
- no multi-document packing is allowed,
- if the data pipeline cannot enforce this, the experiment is blocked until it can.

### 5.6 Training-time execution order

For each TTT-enabled layer and each chunk, the order is strictly:

1. compute frozen pre-output features `Z_c`,
2. apply current `W_0 + DeltaW_c`,
3. continue the model forward pass with that output,
4. compute `Vhat_c`,
5. update `DeltaW_{c+1}`.

This is **apply then update**.

Do not update before applying.

### 5.7 Reference implementation first

Before any optimized implementation:

- implement a **plain sequential chunk loop**,
- verify correctness with unit tests,
- only then consider scan/prefix-sum acceleration.

---

## 6. Main experimental plan

## Phase 0 - correctness and unit tests

This phase is mandatory. Do not begin main training without passing all items.

### TODO 0.1 - expose block internals

- [ ] expose `pre_out_features` for one SSM block
- [ ] verify shape `[B, T, m_l]`
- [ ] verify replacing `out_proj` with an explicit matrix multiply reproduces the original output

### TODO 0.2 - zero-update identity test

- [ ] set all target-builder parameters to zero
- [ ] run the TTT-enabled model and vanilla model on the same batch
- [ ] verify logits match to numerical tolerance

Required threshold:

- fp32: max absolute difference `< 1e-6`
- bf16/fp16: relative tolerance comparable to a standard mixed-precision forward

### TODO 0.3 - chunk-equivalence test

- [ ] set `DeltaW = 0`
- [ ] run with chunk size `C = T`
- [ ] run with chunk size `C = 256`
- [ ] verify outputs match when target-builder parameters are zero

### TODO 0.4 - boundary-isolation test

- [ ] construct a toy sequence with two chunks
- [ ] manually change tokens in chunk 2
- [ ] verify `Vhat` for chunk 1 is unchanged

### TODO 0.5 - reset test

- [ ] feed document A then document B in one batch item with an explicit reset between them
- [ ] verify outputs on document B match a fresh run on document B alone

### TODO 0.6 - toy repeated-pattern test

Use a tiny synthetic task such as repeated key-value or repeated n-gram continuation.

- [ ] confirm that one adapted layer with the exact next-token target reduces loss on repeated patterns faster than the vanilla SSM
- [ ] log `||DeltaW||_F` and `||G||_F`

This is only a sanity test. Do not over-interpret it.

---

## Phase 1 - tiny pilot

Goal: establish that the implementation trains stably and the TTT path is actually used.

### Model scale

Use a very small model first:

- target parameter count: `50M-150M`
- context length: `2048` or `4096`
- token budget: `0.5B-1B`

### Models to train

Train exactly these three:

1. `Transformer-SWA`
2. `Vanilla-SSM`
3. `SSM + In-Place TTT`

### Fixed settings

- same tokenizer
- same training data subset
- same optimizer family
- same global batch size in tokens
- same sequence length
- same token budget

### TTT settings for the tiny pilot

- target source: token embeddings
- `K_tgt = 5`
- chunk size `C = 128`
- TTT-enabled layers: 4 evenly spaced layers if depth allows; otherwise 2 evenly spaced layers
- clipping: off

### Required logs

Per training step or logging interval record:

- LM loss
- validation perplexity at 2k and 4k if available
- mean `||G||_F` per TTT layer
- mean `||DeltaW||_F` per TTT layer
- fraction of chunks with `||G||_F == 0`
- throughput and peak memory

### Exit criterion for Phase 1

Proceed only if:

1. training is stable for all three models,
2. `SSM + In-Place TTT` shows non-trivial fast-weight activity,
3. there is no evidence that the TTT path is numerically dead.

If the TTT path stays effectively zero, stop and debug initialization or precision.

---

## Phase 2 - medium pilot

Goal: get the first real long-context signal before the expensive main run.

### Default configuration

- target parameter count: `300M-500M`
- training context length: `16384` if 32k is too expensive; otherwise `32768`
- token budget: `2B-5B` minimum, more if affordable

### Models

Again train exactly these three:

1. `Transformer-SWA`
2. `Vanilla-SSM`
3. `SSM + In-Place TTT`

### Default TTT settings

- target source: token embeddings
- `K_tgt = 5`
- chunk size `C = 256`
- TTT layers: 4 evenly spaced layers
- clipping: off unless instability occurs

### Mandatory ablations in Phase 2

Only these three ablations are allowed before the main run:

1. chunk size `C in {128, 256, 512}`
2. number of TTT-enabled layers in `{2, 4, 8}`
3. target type:
   - exact next-token target
   - learned `K_tgt = 5` target

Do **not** open a broad ablation zoo.

### Exit criterion for Phase 2

Proceed to the main run only if one of the medium-scale TTT configurations:

- beats `Vanilla-SSM` at `8k+`, and
- is at least competitive with or better than `Transformer-SWA` at `8k+`.

If this does not happen, do not start the expensive run. First debug or pivot.

---

## Phase 3 - main Figure-2 deliverable

This is the main deliverable the agent must produce.

### Preferred configuration

Use a 500M-like matched setting if compute allows.

#### Transformer-SWA reference configuration

Mirror the paper’s 500M-style reference as closely as possible for the Transformer baseline:

- `d_model = 1024`
- `num_layers = 24`
- `d_ff = 3072`
- `num_heads = 8`
- SWA window = `2048`
- sequence length = `32768`

#### SSM and SSM+TTT configuration

Choose SSM width/depth so that total parameter count is within **±5%** of the Transformer baseline.

Do not match hidden size blindly. Match **actual parameter count**.

### Training defaults for the main run

If resources permit, use this target setup:

- sequence length: `32768`
- token budget: `20B`
- optimizer: `AdamW`
- learning rate: `5e-4` for the 500M-like run
- weight decay: `0.1`
- gradient clipping on outer optimizer: `1.0`
- warmup: `1024` optimizer steps
- tokenizer: one shared tokenizer across all three models

If the exact token budget is infeasible, do not claim a final Figure-2 reproduction. In that case label the output clearly as a **pilot figure**.

### Main TTT defaults for the main run

- target source: token embeddings
- target kernel size: `5`
- chunk size: start from the best Phase-2 value, default `256`
- TTT-enabled layers: 4 evenly spaced layers
- fast-weight math: exactly as in Section 4 above
- clipping: off unless needed for stability

---

## 7. Figure-2-style evaluation spec

This section must be followed exactly.

### 7.1 Validation set

Use a public held-out LM validation set. Preferred:

1. `The Pile` validation split
2. optionally also `Proof-Pile-2` for an auxiliary plot

The first deliverable only requires one main plot; use the same validation set for all models.

### 7.2 Context lengths

Evaluate at:

- `2k`
- `4k`
- `8k`
- `16k`
- `32k`

### 7.3 Scored suffix

For a 500M-like model, score the **final 2048 tokens**.

For larger models, score the **final 4096 tokens**.

### 7.4 Exact evaluation protocol

For each validation segment of length `32768`:

1. fix a scored suffix consisting of the **last 2048 tokens**,
2. for each context length `L in {2k, 4k, 8k, 16k, 32k}` with `L >= 2048`, provide only the **last L tokens** of the segment to the model,
3. compute NLL/perplexity **only on the fixed final 2048-token suffix**,
4. average over all validation segments.

This means the predicted tokens are the same across context lengths; only the amount of available left context changes.

### 7.5 Plot specification

Produce one line plot with:

- x-axis: context length
- y-axis: perplexity
- one curve for each of:
  - `Transformer-SWA`
  - `Vanilla-SSM`
  - `SSM + In-Place TTT`

Optional additional curves are allowed only after the required three are present.

### 7.6 Required textual interpretation

The report accompanying the plot must answer exactly these questions:

1. Does `SSM + In-Place TTT` beat `Vanilla-SSM` at long context?
2. Does `SSM + In-Place TTT` beat `Transformer-SWA` at long context?
3. At what context length does the gap first appear?
4. Is the gap widening, flat, or closing as context grows?

---

## 8. Required baselines and fairness rules

## Required models

The main run must include exactly these:

1. **Transformer-SWA**
2. **Vanilla-SSM**
3. **SSM + In-Place TTT**

## Fairness rules

All three must share:

- tokenizer
- training data
- validation data
- optimizer family
- sequence length
- token budget
- batch size in tokens, as closely as memory allows
- comparable parameter count

Do not compare models trained under different token budgets or different corpora and call it a clean result.

---

## 9. Mandatory diagnostics

The agent must produce the following diagnostics in addition to the main plot.

### 9.1 Fast-weight health checks

For every TTT-enabled layer, log:

- mean `||G||_F`
- mean `||DeltaW||_F`
- max `||DeltaW||_F`
- histogram or percentile summary of `||G||_F`
- fraction of chunks where `G` is exactly zero or numerically tiny

### 9.2 Ablation diagnostics

At minimum provide:

1. `Vanilla-SSM` vs `SSM + exact next-token target`
2. `SSM + exact next-token target` vs `SSM + learned K=5 target`
3. chunk size comparison for `128/256/512`

### 9.3 Efficiency diagnostics

Record:

- training throughput in tokens/s
- evaluation throughput in tokens/s
- peak memory

Do not spend more than one short run on efficiency until the quality signal exists.

---

## 10. Required deliverables

The agent must produce the following artifacts.

### Deliverable A - code

A clean implementation of:

- TTT-enabled SSM block wrapper
- target builder
- sequential chunk apply-then-update loop
- evaluation script for sliding-window perplexity

### Deliverable B - correctness note

A short note or markdown section showing that Phase-0 tests passed.

### Deliverable C - pilot results

At least one medium-scale pilot plot with the three required curves.

### Deliverable D - main Figure-2-style plot

A publication-quality plot with:

- context length on x-axis
- perplexity on y-axis
- three required curves
- model sizes and training budget stated in the caption

### Deliverable E - experiment summary table

A table with one row per trained model containing:

- architecture
- parameter count
- sequence length
- token budget
- chunk size
- number of TTT layers
- final validation perplexity at each context length

### Deliverable F - conclusion memo

A short memo answering:

1. Did the method work?
2. Was the gain over SWA real?
3. What was the best chunk size?
4. What was the best number of TTT-enabled layers?
5. What should the next experiment be?

---

## 11. Failure modes and decision tree

### Failure mode 1 - TTT path is dead

Symptoms:

- `||G||_F` near zero
- `||DeltaW||_F` near zero
- no difference from vanilla SSM

Actions:

1. verify target builder is not still zero-initialized after training starts,
2. verify `W_tgt` and target-builder params are in fp32,
3. verify gradients reach target-builder params,
4. verify chunk boundary masking is not zeroing everything.

### Failure mode 2 - instability / NaNs

Symptoms:

- exploding `||DeltaW||_F`
- sudden loss spikes
- eval NaNs

Actions:

1. turn on fast-weight update clipping with `tau = 1.0`,
2. verify `G` uses the **mean** over chunk, not the sum,
3. ensure `DeltaW` is stored in fp32,
4. reduce chunk size from `512` to `256` or `128`.

### Failure mode 3 - improves over vanilla SSM but not SWA

Actions, in order:

1. tune chunk size over `{128, 256, 512}`
2. tune number of TTT layers over `{2, 4, 8}`
3. compare exact next-token target vs learned `K=5` target
4. increase token budget if the curves are still moving downward at the end of training

Only after these steps may you pivot the architecture.

### Failure mode 4 - pure SSM does not respond well to TTT even after targeted tuning

Fallback plan:

- keep the exact same TTT math,
- move to a **hybrid SSM + local-attention** backbone,
- still adapt only the block output projection of the selected SSM-like blocks.

Do not jump to this fallback until the pure-SSM plan has been given a fair attempt.

---

## 12. Exact ablation order

Run ablations in this order and stop as soon as a clear winner emerges.

1. `C = 128` vs `C = 256`
2. winner vs `C = 512`
3. `4 TTT layers` vs `2 TTT layers`
4. winner vs `8 TTT layers`
5. exact next-token target vs learned `K=5` target

Do not run broad combinatorial sweeps in the first pass.

---

## 13. Later extension: continual-training / warm-start track

This is **not** the first deliverable, but it should be the next track after the main figure works.

For a warm-started pretrained SSM:

- keep the same fast-weight location,
- keep the same chunked apply-then-update math,
- replace target source `q_t = e_t` with the **layer input hidden state** `q_t^l = h_t^l`,
- keep the same learned `K=5` future-mixing target builder,
- consider turning on fast-weight clipping during long-context evaluation.

This matches the continual-training flavor of the appended paper more closely.

---

## 14. Pseudocode for the reference implementation

```python
# Inputs:
# tokens: [B, T]
# ttt_layers: selected layer indices
# chunk_size = C

x = embed(tokens)                     # [B, T, d]
source_embeddings = x.detach() or x   # for from-scratch target source use embeddings

for l, block in enumerate(model.blocks):
    if l not in ttt_layers:
        x = block(x)
        continue

    # 1) frozen block internals up to pre-output features
    h_in = x                          # [B, T, d]
    z = block.pre_out_features(h_in)  # [B, T, m_l]
    W0 = block.out_proj_weight        # [d, m_l]

    # 2) build target source for this layer
    q = source_embeddings             # [B, T, d] for main figure

    # 3) build future-looking LM-aligned target with within-chunk boundary zeros
    vhat = build_target(q, kernel_size=5, chunk_size=C)  # [B, T, d]

    # 4) chunk loop
    deltaW = zeros([B, d, m_l], dtype=float32, device=x.device)
    outputs = []
    for s in range(0, T, C):
        e = min(s + C, T)
        zc = z[:, s:e, :]             # [B, C, m_l]
        vc = vhat[:, s:e, :]          # [B, C, d]

        # apply current fast weights
        W_eff = W0[None, :, :].float() + deltaW            # [B, d, m_l]
        oc = einsum("bdm,bcm->bcd", W_eff, zc.float())    # [B, C, d]
        outputs.append(oc.to(x.dtype))

        # update after apply
        G = einsum("bcd,bcm->bdm", vc.float(), zc.float()) / (e - s)
        # optional clip if needed:
        # G = frobenius_clip(G, tau=1.0)
        deltaW = deltaW + G

    o = concat(outputs, dim=1)        # [B, T, d]
    x = block.post_out_combine(h_in, o)

logits = lm_head(final_norm(x))
loss = next_token_cross_entropy(logits, tokens)
```

### Important pseudocode notes

1. `build_target` must zero out future offsets that cross a chunk or document boundary.
2. `deltaW` is per batch item.
3. `deltaW` is reset at every document boundary.
4. The outer LM loss is standard cross-entropy.
5. The reference implementation is intentionally sequential.

---

## 15. Minimum report template

The final report for the first deliverable must contain the following headings:

1. `Goal`
2. `Models compared`
3. `Exact TTT math`
4. `Training setup`
5. `Evaluation setup`
6. `Figure-2-style result`
7. `Ablations`
8. `Diagnostics`
9. `Conclusion`
10. `Next step`

---

## 16. Final decision rule

At the end of the first campaign, make exactly one of these calls:

### Outcome A - success

Use this only if `SSM + In-Place TTT` clearly beats both `Vanilla-SSM` and `Transformer-SWA` at long context.

### Outcome B - partial success

Use this if it clearly beats `Vanilla-SSM` but is still slightly behind `Transformer-SWA`.

Then the next experiment is:

- tune chunk size and TTT layer count once more,
- then try the hybrid SSM + local-attention fallback if needed.

### Outcome C - failure of the pure-SSM first attempt

Use this only if there is no convincing gain over `Vanilla-SSM` after the targeted ablations.

Then pivot to the hybrid fallback while keeping the exact same TTT math.

---

## 17. One-paragraph summary for an agent

Implement in-place TTT on a Mamba-like SSM by adapting only selected blocks’ `out_proj` matrices. Use token embeddings as the target source for the first from-scratch Figure-2-style experiment. Build an LM-aligned future-looking target with kernel size 5 and a trainable projection. Run a sequential chunkwise apply-then-update loop with per-sequence fast weights, resetting at document boundaries. Train three matched models—SWA Transformer, vanilla SSM, and SSM+TTT—under the same tokenizer/data/optimizer/token budget. Evaluate sliding-window perplexity on a fixed final 2048-token suffix while varying available left context over `{2k,4k,8k,16k,32k}`. The first deliverable succeeds only if SSM+TTT beats both vanilla SSM and SWA at long context.

---

## 18. Source basis for this plan

This plan is based on the appended paper’s core choices:

- in-place adaptation of the final projection matrix,
- LM-aligned future-token target,
- chunkwise apply-then-update execution,
- large-chunk ablation logic,
- Figure-2-style sliding-window perplexity evaluation,
- initialization of the new target-building modules so the TTT path starts near zero.

