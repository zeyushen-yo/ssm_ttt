# SSM + In-Place TTT — Updated Spec v4

## 0. Purpose of this document

This document replaces the earlier v2/v3 specs.

It is written for an implementation agent. The goal is to diagnose why the current SSM + TTT system gives only a small, mostly context-independent gain over vanilla SSM and still trails SWA, and then to execute a concrete next plan that stays as close as possible to **strict in-place TTT**.

Primary constraint:
- Keep the model architecture unchanged in the main line.
- Prefer strict in-place adaptation of an existing projection (`W_eff = W0 + DeltaW`) over adding new modules.
- Treat residual fast path as backup only, not primary.

Primary research goal:
- Make the TTT gain clearly **context-dependent**.
- Ideally reduce or close the gap to SWA.
- If that is not achievable on generic Pile perplexity, determine whether the limitation is in the current update rule, the training data/regime, or pure SSM expressivity.

---

## 1. Current status

### 1.1 What the current code is doing

The current implementation is not a toy anymore. It is a serious and mostly correct implementation of the previous idea:
- It wraps selected Mamba2 blocks and adapts the block `out_proj` in place.
- It performs chunkwise **apply-then-update**.
- It builds LM-aligned future-looking targets with a depthwise combiner plus `W_tgt`.
- It has normalized EMA-style fast-weight updates, Frobenius caps, optional boundary-aware masking/reset, and a 4-group optimizer split.

### 1.2 What the current committed results say

The committed report supports the following picture:
- In the fair 32k run, SWA is best, vanilla SSM is worse, and the best current TTT config improves vanilla SSM only modestly.
- Stage 1 C2 (chunk 64) is the best currently committed pure in-place TTT config at scale.
- The gain over vanilla SSM is real, but small and roughly flat with context length.
- The report itself explicitly raises the same concern: the improvement does not widen with context and may partly be a parameter-count effect.

### 1.3 My interpretation

The implementation is **mostly correct**. The main problem is no longer a catastrophic bug. The current issue is that the **current write rule is likely the wrong memory rule for pure SSMs**.

The current TTT path looks like a small learned correction that helps on average, but it does **not** yet look like a mechanism that stores more useful information as more context arrives.

---

## 2. Main diagnosis

## 2.1 What is probably *not* the main issue

It is probably **not** the case that the current repo simply fails to implement the previous idea.

The current implementation does capture the previous idea reasonably faithfully:
- selected Mamba2 `out_proj` weights are adapted online;
- updates are chunkwise and causal;
- targets are future-looking;
- boundary-aware masking/reset exists;
- previous padding and worker-randomness bugs were addressed.

So this is not mainly “the code forgot to do TTT.”

## 2.2 What *is* the main issue

The main issue is the **nature of the fast-weight update**.

The current rule is effectively Hebbian / similarity-style:

```math
G_c \propto \hat V_c^\top Z_c
```

with normalized EMA accumulation:

```math
\Delta W_{c+1} = \lambda \Delta W_c + (1-\lambda)\,\eta\,G_c.
```

This has a specific failure mode on natural language corpora:
- every chunk writes;
- common/predictable structure writes just as eagerly as rare/useful structure;
- after a few thousand tokens, the fast state approaches a stationary average correction;
- the correction then acts like a small global bias shift rather than a context-growing associative memory.

This exactly matches the current empirical pattern:
- `deltaW_rel` grows a bit, then saturates;
- perplexity improves a bit, then stays flat with additional context;
- longer decay can look good in a tiny run but hurt at larger scale because it preserves more average drift.

This is the core hypothesis for why the current gain is small and not context dependent.

## 2.3 Why this is especially plausible in a pure SSM

The original In-Place TTT paper’s large-chunk success relies on leaving **attention intact** while adapting an existing MLP projection. In a pure SSM backbone there is no attention safety net doing precise token-to-token routing. That means the fast-weight path must do more of the context-specific work itself, and a dense Hebbian write rule is much more likely to wash into a stationary average. In the paper, the authors also explicitly say their framework is orthogonal to the choice of loss/optimizer, which leaves room to modify the write rule for the SSM setting. fileciteturn3file0

---

## 3. Concrete code-level issues that still matter

These are not the main scientific issue, but they should be fixed.

### 3.1 Boundary-aware reset only works for batch size 1

In the wrapper, boundary-aware subspan logic only runs when `segment_ids is not None and B == 1`.

Implication:
- boundary-aware training is silently unsupported for `batch_size > 1`;
- any future run that tries larger batch with true boundary semantics will not actually test the intended algorithm unless this is fixed.

### 3.2 Best current 32k config trains across unrelated document boundaries

The best committed 32k TTT config uses packed documents without boundary awareness. That means the fast weights are trained across unrelated document boundaries.

Implication:
- this departs from the intended “reset at document boundary” semantics;
- it may teach the TTT path to become a generic global correction rather than a document-specific memory.

### 3.3 Current chunk normalization likely under-scales writes

The update uses

```math
G_c = \frac{\hat V_c^\top Z_c}{C}
```

(or the equivalent with normalized inputs). With normalized token features, that makes update magnitude decrease with chunk size. That likely explains why chunk 64 beats chunk 128 partly because of scale, not just because of granularity.

### 3.4 Missing evaluation control: same checkpoint with TTT updates disabled

Right now there is no committed “TTT-on vs TTT-off using the same trained checkpoint” evaluation.

This is required to separate:
- extra parameter effect;
- changed training dynamics;
- actual benefit from online adaptation.

### 3.5 Minor code hygiene item

`train.py` uses `sys.exit(0)` in the preemption path but does not import `sys`.

---

## 4. Updated scientific hypothesis

### 4.1 Hypothesis H1 — current rule stores mean correction, not context-specific memory

The current Hebbian update stores average predictive correlations from recent chunks. On a broad corpus this converges to a stationary correction.

Prediction:
- TTT-on vs TTT-off gap will be roughly constant with context length.
- Replacing the true prefix with a random prefix of the same length will preserve much of the current TTT gain.

### 4.2 Hypothesis H2 — pure SSM needs an error-corrective write rule

For pure SSMs, strict in-place `out_proj` adaptation can work, but the write rule should be **error-corrective** rather than pure similarity accumulation.

Prediction:
- switching to a delta-rule / online-regression update should reduce the “uniform shift” behavior;
- the TTT-on vs TTT-off gap should widen with context on true prefixes;
- the gain should shrink strongly under random-prefix or shuffled-prefix controls.

### 4.3 Hypothesis H3 — data/training regime suppresses genuine boundary-aware long-memory behavior

Packed random-document training without resets is likely suppressing context-specific memory learning.

Prediction:
- truly boundary-aware training on long-document data (or within-document packing) should improve context dependence more than random packed training, even if raw short-context PPL initially worsens.

---

## 5. Primary algorithmic change: switch to error-corrective in-place TTT

This is the main recommendation.

Keep:
- same SSM backbone;
- same selected `out_proj` fast weights;
- same target builder at first;
- same strict in-place structure.

Change only the **write rule**.

## 5.1 Current baseline rule (for reference)

For chunk `c`:

```math
O_c = Z_c (W_0 + \Delta W_c)^\top
```

```math
G_c^{\text{hebb}} = \frac{\hat V_c^\top Z_c}{C}
```

```math
\Delta W_{c+1} = \Pi_\rho\Big(\lambda \Delta W_c + (1-\lambda)\eta G_c^{\text{hebb}}\Big)
```

where `Pi_rho` is the Frobenius projection.

## 5.2 New primary rule: delta-current

For chunk `c`, first do the usual apply step:

```math
O_c = Z_c (W_0 + \Delta W_c)^\top
```

Then define the error target:

```math
E_c = \hat V_c - \operatorname{sg}(O_c)
```

where `sg` means stop-gradient for the update path.

Now build the update from **prediction residual**, not from raw target alone.

Recommended normalized version:

```math
\tilde Z_c = \operatorname{RMSNorm}(Z_c)
```

```math
G_c^{\text{delta}} = \frac{E_c^\top \tilde Z_c}{\sqrt{C}}
```

and update:

```math
\Delta W_{c+1} = \Pi_\rho\Big(\lambda \Delta W_c + (1-\lambda)\eta G_c^{\text{delta}}\Big)
```

Important implementation notes:
- normalize `Z`, not the error magnitude;
- keep the current `G_rel_cap` and `deltaW_rel_cap`;
- use `sqrt(C)`, not `C`, as the default chunk scaling;
- do **not** use residual fast path in the primary branch.

## 5.3 Why this is the right change

The current rule writes every chunk equally. The delta rule writes only what is **not already predicted** by the current slow+fast state.

That should:
- suppress stationary mean drift;
- reduce uniform context-independent bias shift;
- make the fast state track context-specific residual information;
- make longer context useful only when it contains additional non-redundant signal.

This is still fully in-place.

## 5.4 Secondary variant: delta-base

Also implement:

```math
O_c^{\text{base}} = Z_c W_0^\top
```

```math
E_c^{\text{base}} = \hat V_c - \operatorname{sg}(O_c^{\text{base}})
```

```math
G_c^{\text{delta-base}} = \frac{(E_c^{\text{base}})^\top \tilde Z_c}{\sqrt{C}}
```

This variant makes the fast state explicitly store residual corrections to slow weights only.

Expected ranking:
- `delta-current` best;
- `delta-base` second;
- current Hebbian rule worst for context dependence.

## 5.5 Backup if delta rule alone is still too flat: centered updates

If delta-current still gives a constant shift, add a running update mean:

```math
M_{c+1} = \beta M_c + (1-\beta) G_c
```

```math
\Delta W_{c+1} = \Pi_\rho\Big(\lambda \Delta W_c + (1-\lambda)\eta (G_c - M_c)\Big)
```

Interpretation:
- store deviations from typical recent updates, not the mean update itself.

This is still in-place, but is a second-stage change, not the first one.

---

## 6. Optional selectivity mechanism, still in-place

If needed, add chunkwise surprise gating.

### 6.1 Gate definition

Let

```math
g_c = \operatorname{clip}\left(\frac{1}{C}\sum_{t \in c} \operatorname{rms}(E_t),\ 0,\ g_{\max}\right)
```

Then use

```math
\Delta W_{c+1} = \Pi_\rho\Big(\lambda \Delta W_c + (1-\lambda)\eta g_c G_c\Big)
```

Interpretation:
- highly predictable chunks write little;
- surprising chunks write more.

This is still not a new module. It is just a scalar write gate derived from the existing residual.

### 6.2 Priority

Do **not** start with surprise gating.

Start with delta-current alone.

Add gating only if:
- TTT-on vs TTT-off still looks almost constant with context;
- random-prefix control still shows most of the gain.

---

## 7. Keep the target builder initially, but change how it is used

Do **not** change the target builder first.

Keep the current target builder:

```math
\hat v_t = \left(\sum_{j=1}^{K} d_j \odot q_{t+j}\right) W_{\text{tgt}}
```

with chunk/document boundary masking.

Reason:
- the repo already showed that target-source changes alone are not the dominant lever;
- the current problem is not “future target missing,” it is “write rule stores the wrong thing.”

Only after delta-current is tested should you try:
- shared `W_tgt` across TTT layers;
- diagonal-only or low-rank `W_tgt` to reduce extra parameter confound;
- layer-input source for top-half TTT layers only.

---

## 8. Updated evaluation protocol: add the diagnostics that actually answer the concern

The current Figure-2-style PPL plot is necessary but not sufficient.

Add the following required diagnostics.

## 8.1 Same-checkpoint TTT-on vs TTT-off

For every trained TTT checkpoint, evaluate two modes:
- **TTT-on**: current behavior.
- **TTT-off**: identical checkpoint, but force `DeltaW = 0` for every chunk.

This is the cleanest measure of true online-adaptation benefit.

Required plot:
- `PPL_on(L)` and `PPL_off(L)` for `L in {2k,4k,8k,16k,32k,64k,128k}` when available.
- Also plot `Gain(L) = PPL_off(L) - PPL_on(L)`.

Primary success condition:
- `Gain(32k)` must exceed `Gain(2k)` by a meaningful margin.

## 8.2 Random-prefix control

For each validation document and each context length `L`:
- keep the final scored suffix fixed;
- replace the prefix of length `L - suffix_len` with a random prefix from another document of the same length.

Measure:
- `PPL_true_prefix(L)`
- `PPL_random_prefix(L)`

Interpretation:
- if TTT gain survives random prefix, the mechanism is not using specific context;
- if the modified TTT works, true-prefix should beat random-prefix by a larger margin than vanilla SSM.

## 8.3 Shuffled-prefix control

For each document, keep the same tokens but shuffle chunk order in the prefix.

Interpretation:
- if gain is order-sensitive, the mechanism is using sequential structure;
- if gain survives shuffling, it is probably just adapting to local token statistics.

## 8.4 `deltaW` health diagnostics

Continue reporting:
- `deltaW_rel_mean`
- `G_rel_mean`
- `effective_window_tokens`

Add:
- `ttt_onoff_gain`
- `true_vs_random_prefix_gap`
- `true_vs_shuffled_prefix_gap`
- if surprise gating is used: `write_gate_mean`, `write_gate_p90`

---

## 9. Updated file-by-file implementation plan

## 9.1 `models/ttt_wrapper.py`

### Required changes

Add config flags:
- `update_rule: "hebb" | "delta_current" | "delta_base"`
- `scale_mode: "mean" | "sqrt_len" | "sum"`
- `normalize_z: bool`
- `normalize_err: bool` (default false)
- `disable_updates: bool` (for evaluation)
- `write_gate: "none" | "chunk_err"`
- `write_gate_max: float`
- `center_updates: bool`
- `center_beta: float`

### Exact code changes

1. Split current `_update_deltaW(...)` into:
- `compute_update_matrix(...)`
- `apply_update(...)`

2. In the chunk loop, compute `o_sub` first, then pass it into update computation.

3. Implement three update rules:
- `hebb`
- `delta_current`
- `delta_base`

4. Implement three scale modes:
- `mean`: divide by `span_len`
- `sqrt_len`: divide by `sqrt(span_len)`
- `sum`: no division

5. Add `disable_updates` so evaluation can compare the same checkpoint with updates on vs off.

6. Boundary-aware path:
- remove the silent `B == 1` assumption;
- either vectorize boundary handling for `B > 1`, or immediately raise an assertion if `segment_ids is not None and B > 1`.
- Do **not** silently fall back to no resets.

### Default values for the new primary branch

Use these defaults:
- `update_rule: delta_current`
- `scale_mode: sqrt_len`
- `normalize_z: true`
- `normalize_err: false`
- `disable_updates: false`
- `write_gate: none`
- `center_updates: false`

## 9.2 `models/ssm_ttt_model.py`

### Required changes

1. Thread the new wrapper config through model creation.
2. Add an optional `shared_target_builder` mode.
3. Allow explicit top-heavy layer placement without code edits.

### New config options

- `ttt_update_rule`
- `ttt_scale_mode`
- `ttt_normalize_z`
- `ttt_normalize_err`
- `ttt_disable_updates`
- `ttt_write_gate`
- `ttt_write_gate_max`
- `ttt_center_updates`
- `ttt_center_beta`
- `ttt_shared_target_builder`

### Shared target builder option

If `ttt_shared_target_builder: true`:
- instantiate one `TargetBuilder` and hand references to all TTT layers.

Reason:
- reduce extra parameter confound;
- make 6-layer top-heavy TTT affordable without adding much capacity.

## 9.3 `models/target_builder.py`

### Required changes

Keep current behavior as default.

Add optional modes later only if needed:
- `w_tgt_mode: full | diagonal | low_rank`
- `low_rank_rank: int`

Do **not** start with these changes.

## 9.4 `data/dataloader.py`

### Required changes

1. If `boundary_aware: true` and effective batch handling would break resets, make that explicit.
2. Add a new dataset mode for **long-document-only** or **within-document packing**.

### New dataset modes

- `pack_documents: true` (existing)
- `boundary_aware: true` (existing)
- `long_doc_only_min_len: int | null`
- `within_document_packing: true | false`

### Principle

The primary long-context TTT runs should **not** train the fast weights across unrelated document boundaries in the main scientific comparison.

## 9.5 `train.py`

### Required changes

1. Import `sys`.
2. Log the new diagnostics.
3. Save the wrapper mode in config.
4. Add optional gradient accumulation only if compute budget allows more total tokens.

### Important note

Do not assume “bigger effective batch” is automatically better at fixed total tokens. Because at 32k context the optimizer-step budget is already tiny. If accumulation is used, total training tokens should be increased proportionally or the run becomes too short in optimizer steps.

## 9.6 `evaluate.py`

### Required changes

Add new flags:
- `--disable_ttt_updates`
- `--random_prefix_control`
- `--shuffle_prefix_control`
- `--max_context_lengths ...`

### Required outputs

For each model checkpoint:
- standard sliding-window PPL;
- same-checkpoint TTT-on vs TTT-off gain;
- true-prefix vs random-prefix gap;
- true-prefix vs shuffled-prefix gap.

## 9.7 `tests/test_phase0.py`

Add new tests:

1. `delta_current` zero-error no-update test.
2. `scale_mode=sqrt_len` chunk-size invariance sanity test.
3. `disable_updates` identity test.
4. `boundary_aware + B>1` explicit failure or correct vectorized behavior.
5. `center_updates` removes mean-drift in a synthetic stationary-G setting.

---

## 10. Experiment matrix

This is the exact order to run.

## 10.1 Phase A — fast screening (20M to 50M tokens)

Purpose:
- test the new write rule quickly;
- answer whether the gain becomes context-dependent;
- avoid wasting 200M-token runs on the wrong mechanism.

### Fixed setup

Use current C2 as base unless otherwise specified:
- 24-layer Mamba2
- `d_model=768`
- 4 TTT layers at current indices `[5, 10, 14, 19]`
- `chunk_size=64`
- target source = embedding
- strict in-place `W_eff = W0 + DeltaW`
- no residual fast path
- `seq_len=32768`
- packed data for now, but record whether boundary-aware is on/off

### Runs

#### A0 — current C2 re-baseline
- current committed C2 settings
- purpose: reproduce current “small flat gain” behavior in the new eval harness

#### A1 — hebb + sqrt_len
- same as A0
- only change: `scale_mode = sqrt_len`
- purpose: test whether chunk-size normalization is currently suppressing writes

#### A2 — delta_current + sqrt_len
- `update_rule = delta_current`
- `scale_mode = sqrt_len`
- `normalize_z = true`
- `normalize_err = false`
- purpose: primary test

#### A3 — delta_base + sqrt_len
- same as A2 but `update_rule = delta_base`
- purpose: determine whether base-residual or current-residual is better

#### A4 — delta_current + surprise gate
- same as A2
- `write_gate = chunk_err`
- `write_gate_max = 3.0`
- purpose: test selectivity only if A2 still looks flat

### Required evaluation for every A-run

At minimum:
- TTT-on vs TTT-off
- true-prefix vs random-prefix
- contexts: `2k, 4k, 8k, 16k, 32k`

### Phase A acceptance criterion

Advance only if **all** are true for one run:
1. `PPL_on(32k) < PPL_off(32k)` by at least `1.0` PPL.
2. `Gain(32k) - Gain(2k) >= 0.5` PPL.
3. `true_prefix_gap(32k) >= 1.0` PPL over random-prefix.
4. `deltaW_rel` does not hit cap early.

If none pass, go to Phase A-backup:
- enable centered updates;
- try chunk 32.

## 10.2 Phase B — make the memory more SSM-friendly, still in-place

Run only after a Phase A winner exists.

### B1 — chunk 32 vs 64 under delta-current

Use the best Phase A rule and compare:
- `chunk=32`
- `chunk=64`

Only change chunk size. Keep scale mode fixed at `sqrt_len`.

### B2 — top-heavy TTT placement

Use the best Phase A rule and compare:
- current 4-layer placement `[5, 10, 14, 19]`
- top-half 4-layer placement `[12, 16, 20, 23]`
- top-heavy 6-layer placement `[10, 13, 16, 19, 21, 23]`

Primary hypothesis:
- top-heavy placement should improve context dependence because upper-layer features are more semantic and more query-like.

### B3 — shared target builder

Use the best placement and compare:
- per-layer `TargetBuilder`
- shared `TargetBuilder`

Primary purpose:
- reduce extra-parameter confound;
- possibly improve learning of the write target.

### B4 — long-doc / boundary-aware data

Run the best B1/B2/B3 model in two data modes:
- packed random docs without resets
- boundary-aware long-doc / within-document packing

Primary purpose:
- test whether genuine document-specific adaptation produces stronger context dependence.

## 10.3 Phase C — full run (200M+ tokens)

Only after a Phase B winner exists.

### Required models

Train all of the following under the **same** regime:
1. SWA baseline
2. vanilla SSM baseline
3. best new TTT model
4. same new TTT model evaluated with updates off

### Required evaluation contexts

`2k, 4k, 8k, 16k, 32k, 64k, 128k` whenever validation docs support them.

### Required plots

1. Standard Figure-2-style sliding-window PPL.
2. TTT-on vs TTT-off gain vs context length.
3. True-prefix vs random-prefix gap vs context length.

---

## 11. Controls that are now mandatory

## 11.1 Same-checkpoint TTT-on vs TTT-off

This is mandatory. It is the cleanest “is online adaptation actually doing anything?” control.

## 11.2 Parameter-count control

One of the following must be done:
- train a parameter-matched wider vanilla SSM, or
- use shared `TargetBuilder` and report the reduced extra-parameter count, or
- both.

## 11.3 Prefix corruption controls

Mandatory:
- random-prefix control
- shuffled-prefix control

Without these, a constant-shift improvement could still be mistaken for genuine long-context use.

---

## 12. Recommended first exact run

This is the first run I recommend now.

### Run name
`phase3_A_delta_current`

### Config
- backbone: same as current C2
- TTT layers: `[5, 10, 14, 19]`
- chunk size: `64`
- target source: `embedding`
- update rule: `delta_current`
- scale mode: `sqrt_len`
- normalize `Z`: `true`
- normalize error: `false`
- residual fast path: `false`
- boundary aware: `false` for the very first screening run only
- seq len: `32768`
- token budget: `50M` if screening, `200M` only after screening success

### Why this run first
- it changes exactly one scientific thing: the write rule;
- it directly tests the main diagnosis;
- it respects the strict in-place preference;
- it avoids mixing algorithmic and dataset changes too early.

### Required eval after this run
- TTT-on vs TTT-off
- true-prefix vs random-prefix
- contexts `2k,4k,8k,16k,32k`

### What counts as success
- the TTT-on vs TTT-off gain grows with context;
- true-prefix clearly beats random-prefix;
- gain over vanilla is no longer a nearly constant shift.

---

## 13. What not to do first

Do **not** do these first:
- residual fast path as the main branch;
- hybrid SSM + attention architecture changes;
- target-source changes alone;
- only longer decay with the current Hebbian rule;
- only larger `G_rel_cap` with the current Hebbian rule.

These do not directly attack the main failure mode.

---

## 14. Decision tree

### Case 1 — delta-current works

If `delta_current` produces context-dependent gain:
- keep strict in-place formulation;
- scale to 200M+;
- then optimize placement, chunk size, and data regime.

### Case 2 — delta-current helps but still trails SWA by a lot

Then the next likely bottleneck is data/training regime, not the core in-place mechanism.

Do next:
- boundary-aware long-doc training;
- top-heavy placement;
- shared target builder;
- evaluate at 64k/128k.

### Case 3 — delta-current still gives constant-shift gain

Then the next change is centered updates or surprise gating.

### Case 4 — even after centered/gated delta rule the gain is still context-independent

Then the out-projection-only in-place memory is likely too weak for pure SSM on this task.

At that point, the next backup should still remain in-place if possible, e.g.:
- top-layer LM-head in-place adaptation as a diagnostic control;
- or a different existing projection inside the SSM stack.

But do not jump there yet.

---

## 15. Bottom line

My updated view is:

1. The current repo is mostly implementing the previous v3 idea correctly.
2. The current small flat gain is probably **not** a hidden implementation bug.
3. The current Hebbian/similarity write rule is the wrong write rule for getting context-growing gains in pure SSMs.
4. The best next move is to keep strict in-place TTT and switch the write rule to an **error-corrective delta rule**, plus add the missing on/off and prefix-corruption controls.
5. Only after that should you decide whether the limitation is the data regime or pure SSM expressivity.

