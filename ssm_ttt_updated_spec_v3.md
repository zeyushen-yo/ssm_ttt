# Updated Spec v3: SSM + In-Place TTT

## Executive summary

This spec replaces the previous working assumption that the current repo is already a faithful test of the core idea.

My current view is:

1. The repo now has a stable v3 implementation, and the long-context explosion problem is largely fixed.
2. The present negative result versus SWA **does not** mean TTT-on-SSM is dead.
3. The current 32k experiment is still **not the right test** of the intended mechanism, because packed training violates TTT boundary assumptions and the learned decay range cannot represent a full 32k memory horizon.
4. Even after those fixes, a pure in-place perturbation of Mamba `out_proj` may still be weaker than SWA on generic within-train-length Pile perplexity. Therefore the plan now has two levels:
   - first, fix the current implementation so it is a correct test of the idea;
   - second, if pure in-place SSM+TTT still trails SWA, move to a stronger SSM-specific formulation: **residual fast path first**, then **projected-key fast memory** or a **hybrid SSM + local attention** model.

The most important concrete change is:

- **make packed training document-boundary aware** for both target construction and fast-weight reset.

The second most important change is:

- **increase the learned forgetting horizon** so it can cover the full 32k training context.

---

## Current diagnosis

### What the repo currently demonstrates

The current repo demonstrates three things:

1. The v3 update rule is stable.
2. SSM+TTT is genuinely helping the vanilla SSM.
3. The current pure SSM+TTT configuration is still worse than the SWA baseline in the fair 32k comparison.

That means the question is no longer “can TTT be made numerically stable on SSMs?”
It can.
The real question is now:

- is the current implementation still mismatched to the intended algorithm?
- or is the pure SSM version of the idea too weak and in need of architectural modification?

My answer is: **both**.

### What I think is wrong, ranked by priority

#### Priority 1 — packed 32k training breaks the intended TTT document semantics

The current 32k runs use packed training sequences that may span document boundaries. That is acceptable for vanilla causal LM training, but it is **not acceptable as-is** for TTT unless the implementation explicitly handles document boundaries.

TTT assumes:

- the future-looking target must not look across a document boundary;
- fast weights must reset at document boundaries.

The current code only masks future mixing across **chunk boundaries**, not document boundaries. It also does not reset the fast weights inside a packed training sequence when a document boundary occurs.

This creates two distinct training-time errors:

1. the target builder can produce targets that mix the end of one document with the beginning of the next;
2. the fast weights can store information from one document and apply it to the next document.

This is exactly the kind of mismatch that hurts TTT more than ordinary next-token training.

#### Priority 2 — the learned decay range is too short for 32k training

The current normalized EMA rule is good, but its allowed decay range is too restrictive for a 32k experiment.

Current config:

- `chunk_size = 64`
- `decay_max = 0.995`
- `decay_init = 0.95`

Under the repo’s own diagnostic definition,

- effective window in tokens = `chunk_size / (1 - decay)`

So:

- at `decay = 0.95`, the effective horizon is only about `64 / 0.05 = 1,280` tokens;
- even at `decay = 0.995`, the maximum horizon is only about `64 / 0.005 = 12,800` tokens.

That is below the full 32k training length.

This matches the report’s observation that `DeltaW_rel` saturates by about 8k and both TTT curves become flat from 8k to 32k.

#### Priority 3 — the TTT target-builder parameters are still being regularized like ordinary weights

The target-builder parameters (`mix_coeffs`, `W_tgt`) start near zero by design.
That is good for stability.
But in the current training loop they are still placed in the standard decayed parameter group.

That is probably too harsh for a small, newly introduced module that is supposed to “emerge” from near-zero.

This likely slows or suppresses the learning of the TTT path.

#### Priority 4 — the stable v3 path may now be too conservative

The current setup appears to have solved instability by making the TTT perturbation very small.
That is better than exploding, but it may now be too weak to compete with SWA.

The important pattern is:

- `DeltaW_rel` is tiny and saturates early;
- TTT beats vanilla SSM, so the mechanism is real;
- but the gain is modest, so the mechanism is not yet strong enough.

#### Priority 5 — the pure “in-place overwrite” form may be suboptimal for pure SSMs

The current wrapper supports two apply rules:

1. strict in-place: `W_eff = W0 + DeltaW`
2. residual fast path: `o = Z W0^T + g * Z DeltaW^T`

For a pure SSM, I now think the residual fast path is the better default.

Reason:

- in a pure Mamba block, `out_proj` is the main readout from the recurrent state;
- perturbing it directly forces the fast update to remain tiny so the base model is not damaged;
- an additive gated fast path lets the base readout stay intact while the TTT memory contributes only when it helps.

This is the closest “change of idea” that still stays true to the original project.

---

## Updated research hypothesis

### Old hypothesis

Adapting Mamba `out_proj` in-place with an LM-aligned fast-weight update should beat SWA for long-context perplexity.

### Updated hypothesis

A **boundary-aware, horizon-matched, residualized** SSM+TTT should improve significantly over vanilla SSM and may beat SWA at long contexts, especially beyond the training length.

A stronger version of the hypothesis is:

- pure in-place `out_proj` adaptation may improve vanilla SSM but may not be the best formulation for beating SWA;
- the best SSM-specific formulation is likely one of:
  1. residual fast path on top of `out_proj`;
  2. projected-key fast memory on top of SSM features;
  3. hybrid SSM + local attention + TTT.

So the project is now split into:

- **Track A:** fix the current pure SSM implementation so it is a valid test;
- **Track B:** strengthen the pure SSM formulation;
- **Track C:** if needed, pivot to projected-key or hybrid backbones.

---

## Exact math to implement now

## Notation

For one TTT-enabled SSM layer `l`:

- `z_t^l in R^{d_inner}` = pre-output SSM feature at token `t`
- `W0^l in R^{d_model x d_inner}` = frozen/base `out_proj`
- `DeltaW_t^l in R^{d_model x d_inner}` = fast weight state
- `q_t^l in R^{d_model}` = source vector for target construction
- `vhat_t^l in R^{d_model}` = LM-aligned target direction
- `seg_t` = document/segment id of token `t`
- `C` = chunk size

### Source vector

For from-scratch training, use:

`q_t^l = E[x_t]`

Do **not** prioritize switching this yet; keep the current embedding-based source until the higher-priority fixes are done.

### Boundary-aware LM target

For kernel size `K`:

`u_t^l = sum_{j=1}^K (d_j^l ⊙ q_{t+j}^l) * 1[seg_{t+j} = seg_t] * 1[chunk(t+j) = chunk(t)]`

`vhat_t^l = u_t^l (W_tgt^l)^T`

This is the current repo target, plus the missing document-boundary mask.

### Normalized update matrix

For a chunk or chunk-subspan `S` contained within a single document segment:

`Z_S in R^{|S| x d_inner}` = stack of pre-output features

`V_S in R^{|S| x d_model}` = stack of target directions

Normalize row-wise along the feature dimension:

`Zbar_S = RMSNorm_lastdim(Z_S)`

`Vbar_S = RMSNorm_lastdim(V_S)`

Then form the chunk update:

`G_S = (Vbar_S^T Zbar_S) / |S|`

Then cap it relative to the base weight norm:

`G_S <- Proj_{||.||_F <= rho_G ||W0||_F}(G_S)`

### Recommended fast-weight state update

Keep the current normalized EMA form, but use a much longer allowed decay range:

`DeltaW_{next} = Proj_{||.||_F <= rho_D ||W0||_F}( lambda * DeltaW + (1 - lambda) * eta * G_S )`

where:

- `lambda in [0.98, 0.9995]`
- initialize near `0.995`
- `rho_G = 0.05`
- `rho_D = 0.10`

### Recommended apply rule: residual fast path

Use this as the new default for pure SSM:

`o_S = Z_S W0^T + g_l * Z_S DeltaW^T`

where `g_l` is a learned scalar gate initialized to `0.01`.

This is preferred over:

`o_S = Z_S (W0 + DeltaW)^T`

for the pure SSM setting.

### Reset rule

At every document boundary, reset the fast state **before applying** to the first token of the new document:

`DeltaW <- 0`

This reset must also occur **inside packed training sequences**.

---

## File-by-file implementation plan

## 1. `data/dataloader.py`

### Goal

Add a packed training dataset that returns document segment information for every token.

### Required change

Implement a new dataset class:

`PackedBoundaryTrainDataset(data_path, offsets_path, seq_len, seed)`

### Required outputs per sample

Return:

- `input_ids: [seq_len]`
- `labels: [seq_len]`
- `segment_ids: [seq_len]`

### Exact behavior

1. Sample a contiguous span `[start, start + seq_len)` from `train.bin`.
2. Use `train_offsets.npy` to determine which original document each token belongs to.
3. Construct `segment_ids` by document index.
4. Return ordinary LM labels, i.e. `labels = input_ids.clone()`.
5. Do **not** mask labels at document boundaries; this remains ordinary packed causal LM training.
6. The boundary information is for the TTT mechanism only.

### Implementation note

Use `np.searchsorted(offsets, positions, side="right") - 1` to map token positions to document ids.

### TODO checklist

- [ ] Add `PackedBoundaryTrainDataset`
- [ ] Add `segment_ids` to returned batch
- [ ] Modify `create_train_dataloader(...)` so `pack_documents=True` selects the boundary-aware packed dataset when `train_offsets.npy` exists
- [ ] Keep existing `DocOffsetTrainDataset` unchanged

---

## 2. `models/target_builder.py`

### Goal

Make future-target construction respect document boundaries as well as chunk boundaries.

### Required API change

Change:

`forward(q, chunk_size)`

to:

`forward(q, chunk_size, segment_ids=None)`

### Exact masking logic

For each shift `j`:

- `cross_chunk = (chunk_of_t != chunk_of_{t+j})`
- `cross_end = (t + j >= T)`
- `cross_doc = (segment_ids[t] != segment_ids[t+j])` if `segment_ids is not None`
- `cross_boundary = cross_chunk OR cross_end OR cross_doc`

Then zero out `q_{t+j}` wherever `cross_boundary` is true.

### TODO checklist

- [ ] Add `segment_ids` argument
- [ ] Add `cross_doc` masking
- [ ] Keep current chunk-boundary masking
- [ ] Add unit test for document-boundary masking

---

## 3. `models/ssm_ttt_model.py`

### Goal

Thread `segment_ids` through the model to each TTT-enabled layer.

### Required API change

Change the forward path to accept:

`forward(input_ids, labels=None, segment_ids=None, **kwargs)`

### Exact behavior

- Non-TTT layers ignore `segment_ids`
- TTT layers receive the same `segment_ids`

### TODO checklist

- [ ] Add `segment_ids` to `SSMTTTModel.forward`
- [ ] Pass `segment_ids` into each `TTTMamba2Block`
- [ ] Keep existing `embedding` and `layer_input` source options intact

---

## 4. `models/ttt_wrapper.py`

### Goal

Implement boundary-aware reset and switch the recommended pure-SSM default to the residual fast path.

### Required API change

Change:

`forward(u, source_embeddings, seq_idx=None)`

to:

`forward(u, source_embeddings, segment_ids=None, seq_idx=None)`

### Required target-builder call

Change:

`vhat = self.target_builder(source_embeddings, chunk_size=C)`

to:

`vhat = self.target_builder(source_embeddings, chunk_size=C, segment_ids=segment_ids)`

### Required boundary-aware update logic

The current chunk loop is not sufficient because a packed chunk may contain multiple documents.

### Exact new behavior per batch item

For each chunk `[s:e)`:

1. Inspect `segment_ids[:, s:e]`
2. Split the chunk into maximal contiguous subspans with constant `segment_id`
3. For each subspan `S` in order:
   - apply current `DeltaW` to `S`
   - compute update `G_S`
   - update `DeltaW`
4. If the next subspan belongs to a different document, reset:
   - `DeltaW[b] = 0`
   before applying to that next subspan

### Important note

Because the 32k runs currently use `batch_size=1`, it is acceptable to implement this with a per-batch-item Python loop first.
Correctness matters more than elegance at this stage.

### New recommended defaults for pure SSM runs

- `use_residual_fast_path = True`
- `decay_factor_init = 0.995`
- `decay_min = 0.98`
- `decay_max = 0.9995`
- `g_rel_cap = 0.05`
- `deltaW_rel_cap = 0.10`
- `inner_lr_init = 0.30`

### TODO checklist

- [ ] Add `segment_ids` argument
- [ ] Add subspan splitting inside packed chunks
- [ ] Reset `DeltaW` at document boundaries
- [ ] Keep current RMS normalization and Frobenius caps
- [ ] Turn on residual fast path in the new recommended config
- [ ] Log `fast_gate` in diagnostics when enabled

---

## 5. `train.py`

### Goal

Stop over-regularizing the target-builder parameters and make the TTT path easier to learn.

### Required optimizer change

Replace the current 2-group optimizer with 4 groups.

### New parameter groups

#### Group A — backbone decayed

Includes ordinary matrix weights in the backbone.

- `lr = base_lr`
- `weight_decay = 0.1`

#### Group B — backbone no decay

Includes biases, norms, embeddings, and ordinary scalar parameters.

- `lr = base_lr`
- `weight_decay = 0.0`

#### Group C — TTT target-builder params

Includes:

- `mix_coeffs`
- `W_tgt`

Use:

- `lr = 3 * base_lr`
- `weight_decay = 0.0`

#### Group D — TTT scalar control params

Includes:

- `log_inner_lr`
- `log_decay_logit`
- `fast_gate`

Use:

- `lr = base_lr`
- `weight_decay = 0.0`

### Optional warm-start schedule

After the boundary-aware code is in place, run this side experiment:

1. initialize from the best vanilla 32k SSM checkpoint;
2. freeze backbone for 10k–20k steps;
3. train only TTT parameters;
4. then unfreeze all weights and continue jointly.

This is an important diagnostic for whether the mechanism works but is simply hard to learn from scratch.

### TODO checklist

- [ ] Add 4 optimizer groups
- [ ] Zero weight decay for `mix_coeffs` and `W_tgt`
- [ ] Add per-group LR multipliers in config
- [ ] Add optional “TTT-only warm-start” schedule

---

## 6. `evaluate.py`

### Goal

Keep the existing Figure-2-style evaluation, but extend it beyond training length.

### Required changes

Keep the current fixed-suffix scoring exactly as-is.
Add extended context lengths:

- `2048`
- `4096`
- `8192`
- `16384`
- `32768`
- `65536`
- `131072`

### Why

If pure SSM+TTT is slightly worse than SWA at 32k but better at 64k or 128k, that is still a meaningful win and changes the interpretation of the project.

### TODO checklist

- [ ] Keep fixed 2048-token suffix scoring
- [ ] Add 64k and 128k contexts
- [ ] Save per-layer diagnostics for all contexts

---

## 7. `tests/test_phase0.py`

Add the following tests.

### Test 1 — document-boundary target masking

Construct a toy packed sequence with two documents inside one chunk.
Verify that `TargetBuilder` never mixes future tokens across the document boundary.

### Test 2 — boundary reset of fast weights

Construct a toy sequence with a document boundary in the middle.
Verify that updates from the first document do not affect outputs on the second document.

### Test 3 — packed dataset returns segment ids

Verify that `PackedBoundaryTrainDataset` returns monotone segment ids and that changes match the offset file.

### Test 4 — residual fast path reduces to base model when gate is zero

Set `fast_gate = 0` and verify outputs match the base `W0` projection exactly.

### TODO checklist

- [ ] Add all 4 tests
- [ ] Run tests before any large training run

---

## What not to change first

This section is important.
Do **not** randomly change everything at once.

### Do not change target source first

The repo already tested a `layer_input` target-source variant in stage 1, and it was worse than the embedding-based default.
So target-source changes are not the first debugging move.

### Do not remove normalized EMA

The v3 normalized discounted update is the reason the model stopped exploding.
Keep it.

### Do not judge the project only at 32k

The right evaluation now includes 64k and 128k.
If the method crosses SWA only beyond the training length, that still matters.

---

## Recommended configs

## Config A — strict reproduction baseline

Purpose: reproduce the current best pure SSM+TTT result from the repo.

- model: current `stage1_32k_C2`
- chunk size: `64`
- TTT layers: `[5, 10, 14, 19]`
- residual fast path: `False`
- decay init/min/max: `0.95 / 0.90 / 0.995`
- train tokens: `200M`
- seq_len: `32768`
- batch size: `1`
- pack documents: `True`

Use this only as the control.

## Config B — corrected pure SSM+TTT default

Purpose: the first corrected test of the idea.

- model: Mamba2 backbone
- TTT layers: `[5, 10, 14, 19]`
- chunk size: `64`
- kernel size: `5`
- target source: `embedding`
- detach source: `False`
- residual fast path: `True`
- inner lr init: `0.30`
- decay init: `0.995`
- decay min: `0.98`
- decay max: `0.9995`
- normalize update: `True`
- norm eps: `1e-6`
- `deltaW_rel_cap = 0.10`
- `G_rel_cap = 0.05`
- pack documents: `True`, **with boundary-aware segment ids and resets**
- optimizer:
  - base LR `6e-4`
  - backbone WD `0.1`
  - target-builder LR multiplier `3x`
  - target-builder WD `0.0`
  - TTT scalar WD `0.0`
- seq_len: `32768`
- effective batch: target `>= 4` via accumulation if hardware requires
- total tokens: `200M` for screening, `400M+` for confirmation

## Config C — corrected fine-grained variant

Purpose: test whether pure SSM wants smaller chunks.

Same as Config B except:

- chunk size: `32`
- decay max: `0.999`

This gives a nominal effective horizon of about 32k tokens at the decay ceiling.

## Config D — stronger upper-layer variant

Purpose: test whether later semantic features are better keys for fast memory.

Same as Config C except:

- num TTT layers: `6`
- layer indices: `[10, 13, 16, 19, 21, 23]`

---

## Experiment plan

## Phase 0 — exact reproduction

### Goal

Reproduce the current repo numbers before changing anything.

### Runs

- SWA 32k baseline
- vanilla SSM 32k baseline
- current C2 32k TTT

### Deliverable

A small markdown note confirming the current fair 32k table matches the repo.

### Success criterion

Numbers are within small noise of the report.

---

## Phase 1 — correctness fixes only

### Goal

Test whether the current gap to SWA is mostly due to boundary-handling and horizon mismatch.

### Runs

All at `seq_len=32768` and `50M` screening tokens:

1. P1-A = current C2 control
2. P1-B = P1-A + boundary-aware packed training only
3. P1-C = P1-B + longer decay range (`0.995 -> 0.9995` max, init `0.995`)
4. P1-D = P1-C + optimizer split for TTT params

### What to measure

At `2k/4k/8k/16k/32k/64k/128k`:

- perplexity
- `deltaW_rel_mean`
- `G_rel_mean`
- `decay`
- `effective_window_tokens`

### Decision rules

- If P1-B improves significantly over P1-A, the boundary bug was a major issue.
- If P1-C keeps improving beyond 8k while P1-A and P1-B flatten, the horizon cap was a major issue.
- If P1-D improves but diagnostics stay small, learning-rate / regularization was a major issue.

---

## Phase 2 — stronger pure SSM formulation

### Goal

Test whether the pure SSM needs the residual formulation to be competitive.

### Runs

All at `seq_len=32768`, `50M`–`100M` screening tokens:

1. P2-A = best Phase-1 config + residual fast path
2. P2-B = P2-A + chunk size `32`
3. P2-C = P2-B + 6 upper-half TTT layers

### Decision rules

- If residual fast path helps without instability, make it the new pure-SSM default.
- If chunk 32 beats chunk 64, treat smaller chunking as the correct pure-SSM regime.
- If upper-half TTT layers help, keep the top-skewed placement.

---

## Phase 3 — fair 32k comparison

### Goal

Run the corrected best pure SSM+TTT against both baselines with more meaningful optimization.

### Runs

All at `seq_len=32768`:

- SWA baseline
- vanilla SSM baseline
- best corrected pure SSM+TTT

### Training budget

- minimum: `400M` tokens
- preferred: `800M` tokens
- effective batch: `>= 4`, preferably `>= 8`

### Primary deliverable

Figure-2-style plot:

- x-axis: context length
- y-axis: sliding-window perplexity
- contexts: `2k, 4k, 8k, 16k, 32k, 64k, 128k`

### Acceptance criteria

At least one of the following must happen:

1. pure SSM+TTT beats SWA at `32k`; or
2. pure SSM+TTT is still slightly worse at `32k`, but beats SWA at `64k+`; or
3. pure SSM+TTT substantially widens the gain over vanilla SSM and shows a clearly better long-context trend.

If none of these happen, do **not** keep iterating the same formulation blindly.
Move to Phase 4.

---

## Phase 4 — pivot if needed

This phase is mandatory if the corrected pure SSM still trails SWA badly.

## Pivot A — projected-key fast memory

### New idea

Do not use raw `z_t` as the associative key.
Instead learn a compact key projection.

### Math

`k_t = LN(z_t) U_k`, where `U_k in R^{d_inner x d_k}` and `d_k in {256, 384}`

Fast state:

`DeltaW in R^{d_model x d_k}`

Apply:

`o_t = W0 z_t + g * DeltaW k_t`

Update:

`G = (Vbar^T Kbar) / |S|`

with the same normalized EMA update as before.

### Why this pivot is promising

It decouples:

- the base SSM readout space
- the associative memory key space

That is likely better than forcing the full raw Mamba pre-output feature to serve as the key.

## Pivot B — hybrid SSM + local attention backbone

### New idea

Keep the TTT mechanism, but no longer demand that a pure SSM beat SWA on generic Pile perplexity.
Instead use a hybrid backbone where local attention handles precise recent token mixing and TTT handles adaptive memory.

### Minimal version

- alternate SSM blocks and local-attention blocks
- keep the same TTT layer design on the MLP/output side

### Why this pivot is promising

The original In-Place TTT paper works well partly because TTT complements attention instead of replacing it.
A pure SSM backbone removes that advantage.
A hybrid backbone restores it.

---

## Diagnostics that must be logged for every TTT run

For each TTT layer and each evaluation context:

- `deltaW_rel_mean`
- `G_rel_mean`
- `decay`
- `effective_window_tokens`
- `inner_lr`
- `fast_gate` if residual path is enabled
- `||mix_coeffs||_F`
- diagonal norm of `W_tgt`
- off-diagonal norm of `W_tgt`

For each training batch:

- number of document boundaries in the sequence
- fraction of chunks containing at least one boundary
- average subspans per chunk after boundary splitting

These logs are not optional. They are necessary to tell whether a run failed because the TTT path was unused, too short, or simply not learning.

---

## Concrete decision thresholds

These thresholds are intentionally explicit so an agent can act without ambiguity.

### Threshold A — horizon sufficiency

By the end of training, for most TTT layers:

- `effective_window_tokens >= 32768`

If not, the model is still not configured to use the full training context.

### Threshold B — no premature saturation

`deltaW_rel_mean` should not become fully flat by `8k` if the goal is to exploit `32k+` context.

A mild increase from `8k` to `32k` is desirable.

### Threshold C — usefulness

At `32k`, best pure SSM+TTT should beat vanilla SSM by at least `5%` relative PPL.

### Threshold D — competitiveness

If best corrected pure SSM+TTT is still worse than SWA by more than `4 PPL` at `32k` **and** does not cross over by `64k`, pivot to Phase 4.

---

## Immediate next actions

Execute these in order.

1. Reproduce the current `stage1_32k_C2` result.
2. Implement `PackedBoundaryTrainDataset` and thread `segment_ids` through the model.
3. Add document-boundary masking in `TargetBuilder`.
4. Add document-boundary resets inside `TTTWrapper`.
5. Change optimizer groups so `mix_coeffs` and `W_tgt` have zero weight decay and higher LR.
6. Run the 4-run Phase-1 screening matrix.
7. If Phase 1 helps, enable residual fast path and run Phase 2.
8. If Phase 2 helps, launch the full 32k comparison at `400M+` tokens.
9. Evaluate at `64k` and `128k` no matter what.
10. If still weak, pivot to projected-key fast memory.

---

## Bottom line

The current repo result should be interpreted as:

- **the mechanism is real**: TTT improves the SSM;
- **the current 32k packed experiment is still mismatched to the intended algorithm**;
- **the original pure in-place formulation is probably not the strongest SSM-specific version**.

So the project should continue, but with a more precise target:

- first make the implementation faithful;
- then make the pure SSM formulation stronger;
- then, if necessary, stop insisting on strict in-place pure SSM and move to residual/projected-key or hybrid designs.

