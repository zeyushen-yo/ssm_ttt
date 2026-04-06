
# Updated Spec for `ssm_ttt`: Diagnosis, Revised Math, and Concrete Execution Plan

## 0. Executive decision

Do **not** conclude that TTT-on-SSM is dead from the current repo state.

The repo already shows a **real positive signal**: the current `SSM + TTT v2` beats both the SWA Transformer and the vanilla SSM at 4k, and beats SWA again at 8k. The long-context failure at 16k/32k is **primarily a fast-weight state scaling problem**, made worse by a short-train / long-eval mismatch, and by a few training-data implementation issues. The current result is better interpreted as:

- the causal chunkwise TTT mechanism is doing something useful;
- the current fast-weight state parameterization is not length-robust enough for pure SSMs;
- a revised formulation should be tested before making any negative conclusion.

The next work should therefore be:

1. **keep** the current chunkwise causal TTT skeleton;
2. **replace** the current `DeltaW = decay * DeltaW + G` rule by a **normalized discounted update**;
3. **bound the fast-weight state relative to the base weight**;
4. **fix the training data pipeline** so sequences are truly single-document and workers do not duplicate RNG;
5. **re-run the cheap pilot** before any medium or large run;
6. only if that still fails, move to a **residual/gated fast path** or a **hybrid SSM+attention** backup.

---

## 1. Audit of the current repo

### 1.1 What is correct and should be kept

The following parts are correct and should be preserved:

- The TTT wrapper is genuinely doing **apply-then-update** chunkwise on the Mamba2 `out_proj`.
- The target builder correctly constructs **future-looking LM-aligned targets** and zeros future shifts that cross a chunk boundary.
- The evaluation script uses the intended **Figure-2-style sliding-window perplexity** protocol: fixed scored suffix, longer and longer left context.
- The positive 4k result is therefore not obviously a measurement artifact.

This means the repo has already validated the right first question: **can fast weights on top of an SSM help when more context is available?** The answer appears to be **yes, at least in the moderate-length regime**.

### 1.2 What is currently wrong or risky

There are six important issues.

#### Issue A — the learned decay update is parameterized in a way that couples memory length and gain

Current repo update:

\[
G_c = \eta \cdot \frac{1}{C} \hat V_c^\top Z_c,
\qquad
\Delta W_{c+1} = \lambda \Delta W_c + G_c
\]

where `lambda = decay` is learned.

This is the single most important problem.

With this rule,

\[
\Delta W_n
=
\sum_{i=1}^{n} \lambda^{n-i} G_i.
\]

If the update statistics are roughly stationary with mean \( \mu_G \), then

\[
\mathbb E[\Delta W_n]
\approx
\frac{1-\lambda^n}{1-\lambda}\mu_G.
\]

So increasing \( \lambda \) does **two things at once**:

- it increases the memory horizon;
- it also multiplies the steady-state magnitude by roughly \( 1/(1-\lambda) \).

This is a bad parameterization for a learned fast memory. It gives the outer optimizer an incentive to drive \( \lambda \to 1 \) whenever “more memory” is useful, because that also amplifies the fast-weight contribution.

This matches the current report: middle TTT layers learned decay values around `0.996–0.997`, which corresponds to an effective scale multiplier on the order of a few hundred.

#### Issue B — the current implementation only clips the *per-chunk update*, not the *cumulative state*

The repo clips `G`, but not `DeltaW` itself.

That means cumulative growth is still uncontrolled:

\[
\|\Delta W_n\|_F
\le
\sum_{i=1}^n \lambda^{n-i}\|G_i\|_F.
\]

So even if each chunk update is individually clipped, the state can still become much larger than the base weight at long contexts.

For pure SSMs, this is especially dangerous because `out_proj` is part of the **core token mixer**, not a side MLP sitting next to attention.

#### Issue C — adapting the full Mamba `out_proj` is more invasive than adapting Transformer `W_down`

In the Transformer paper, the fast weight is the final projection of an MLP block, while attention remains intact and handles local token mixing.

In this repo, the fast weight is the full Mamba mixer `out_proj`. That projection acts on the entire pre-output feature vector of the mixer, including the recurrent/SSM path and any local branch exposed before the final projection. So the TTT state is not just adding memory; it is perturbing the core mixer output map.

This means the current pure-SSM version is **less protected** than the original Transformer setting. The same fast-weight magnitude that is tolerable in a Transformer MLP may be too large in a pure SSM mixer.

#### Issue D — the pilot has a very large sequence-length extrapolation gap

The current pilot trains at:

- `seq_len = 2048`
- `chunk_size = 128`
- so only `16` chunks are seen during training

but evaluates up to:

- `32768` tokens
- so `256` chunks

Even if the update rule were good, that is a large extrapolation gap. With the current unnormalized decay rule, it is almost guaranteed to become unstable.

#### Issue E — the training data pipeline does not actually guarantee single-document sequences

The code claims that the train dataset returns a chunk from a single document, but the implementation samples a random start position in the flat token array, checks only the **first** EOS inside the sampled span, and then slices after that EOS if possible. That does **not** guarantee that the resulting sequence contains exactly one document, and it does not reset TTT inside the sequence.

For TTT, this matters. A fast-weight method is much more sensitive to document boundaries than a plain LM baseline.

#### Issue F — the training dataloader uses per-dataset RNG inside `__getitem__`, ignores `idx`, and does not reseed per worker

This means multiple workers can end up producing highly correlated or even repeated samples, because each worker gets the same dataset object state and `__getitem__` draws random starts internally.

That is a training-quality bug. It may not be the main reason for the 32k blow-up, but it absolutely weakens the reliability of the pilot.

---

## 2. Interpretation of the current results

### 2.1 What the current report already proves

The current report already proves two things:

- the present implementation is **not** just broken at all lengths, because it improves 4k and 8k over SWA and improves 4k over vanilla SSM;
- the current failure mode is strongly **length-dependent**, not uniformly bad.

That makes a pure “the idea does not work” diagnosis too strong.

### 2.2 What the current report does *not* prove

The current report does **not** prove that the current math is adequate.

Specifically, it does not rule out that:

- the 4k gain is coming from useful fast-memory retrieval,
- while the 16k+ failure is caused by fast-weight state scale drift rather than a conceptual limitation.

My current view is that this is exactly what is happening.

### 2.3 Updated conclusion

The right conclusion is:

> The current repo validates the existence of a useful TTT signal on SSMs, but the current fast-weight state dynamics are not yet stable enough for long context. The next step is to fix the state dynamics and the data pipeline, not to abandon the idea.

---

## 3. Revised math to implement next

## 3.1 Mandatory revised state update

Replace the current rule

\[
\Delta W_{c+1} = \lambda \Delta W_c + G_c
\]

with the **normalized discounted rule**

\[
\Delta W_{c+1}
=
\lambda \Delta W_c
+
(1-\lambda)\eta G_c.
\tag{1}
\]

This is the most important change.

Why this form?

Because then

\[
\Delta W_n
=
(1-\lambda)\eta \sum_{i=1}^{n}\lambda^{n-i} G_i,
\]

and the total kernel mass is bounded:

\[
(1-\lambda)\sum_{k=0}^{n-1}\lambda^k
=
1-\lambda^n
\le 1.
\]

So \( \lambda \) controls the **memory window** without automatically multiplying the total gain by \( 1/(1-\lambda) \).

This is the cleanest fix for the current failure mode.

### 3.2 Normalize the update inputs

Before forming the outer product, RMS-normalize both features and targets tokenwise:

\[
\tilde z_t
=
\frac{z_t}{\sqrt{\mathrm{mean}(z_t^2)} + \epsilon},
\qquad
\tilde v_t
=
\frac{\hat v_t}{\sqrt{\mathrm{mean}(\hat v_t^2)} + \epsilon}.
\tag{2}
\]

Then use

\[
G_c
=
\frac{1}{C}\tilde V_c^\top \tilde Z_c.
\tag{3}
\]

This makes the fast update much less sensitive to norm drift across depth, training, or context length.

### 3.3 Add a relative Frobenius cap on the cumulative state

After applying the update, project the cumulative fast weight to a ball relative to the base weight:

\[
\Delta W_{c+1}
\leftarrow
\Pi_{\|\cdot\|_F \le \rho \|W_0\|_F}(\Delta W_{c+1}),
\tag{4}
\]

i.e.

\[
\Delta W_{c+1}
\leftarrow
\Delta W_{c+1}
\cdot
\min\left(
1,
\frac{\rho \|W_0\|_F}{\|\Delta W_{c+1}\|_F + \epsilon}
\right).
\]

Use `rho = 0.10` as the default first value. Sweep `{0.05, 0.10, 0.20}`.

This is mandatory. Per-chunk clipping alone is not enough.

### 3.4 Keep the current chunkwise causal apply/update structure

The chunkwise forward remains:

\[
O_c = Z_c (W_0 + \Delta W_c)^\top + b.
\tag{5}
\]

Then update with equations (1)–(4).

Do **not** change the causal “apply current state, then update after the chunk” logic. That part is already correct.

### 3.5 Bound the decay range

Parameterize

\[
\lambda
=
\lambda_{\min}
+
(\lambda_{\max} - \lambda_{\min}) \sigma(s).
\tag{6}
\]

Default:

- `lambda_min = 0.90`
- `lambda_max = 0.995`

This prevents the optimizer from silently drifting into an effectively unbounded window.

### 3.6 Recommended default target source for the first patched rerun

There are two plausible choices:

- **embedding source**: use token embeddings, but **do not detach**;
- **layer-input source**: use the current layer input \(u_t^l\), RMS-normalized.

For the very first patched rerun, keep the source as **embeddings but without detach** so that the math change is isolated.

Then immediately ablate **layer-input source** on top of the patched math. My current expectation is that pure SSMs may prefer layer-input source, because the fast weight acts on layer-internal features rather than directly on the embedding space.

### 3.7 Optional structural patch if the mandatory patch is still unstable

If equations (1)–(4) still do not stabilize 16k/32k, switch the output to a residual fast path:

\[
O_c = Z_c W_0^\top + g \cdot Z_c \Delta W_c^\top + b,
\tag{7}
\]

with a learnable scalar gate \(g\), initialized small.

This is more conservative than replacing \(W_0\) by \(W_0 + \Delta W\) directly, and is my preferred fallback for pure SSMs.

---

## 4. Exact code changes by file

## 4.1 `models/ttt_wrapper.py`

### Mandatory edits

#### 4.1.1 Add helper functions

Add:

- `rms_norm_lastdim(x, eps=1e-6)`
- `project_fro_rel(deltaW, base_norm, rho, eps=1e-8)`

Use tokenwise RMS normalization for `zc` and `vc`, and relative Frobenius projection for `deltaW`.

#### 4.1.2 Change the state update math

Current behavior:

```python
G = eta * torch.bmm(vc.float().transpose(1, 2), zc.float()) / chunk_len
if self.clip_tau is not None:
    ...
if decay is not None:
    deltaW = decay * deltaW + G
else:
    deltaW = deltaW + G
```

Replace by:

```python
zc_fp32 = zc.float()
vc_fp32 = vc.float()

if self.normalize_update:
    zc_u = rms_norm_lastdim(zc_fp32, eps=self.norm_eps)
    vc_u = rms_norm_lastdim(vc_fp32, eps=self.norm_eps)
else:
    zc_u = zc_fp32
    vc_u = vc_fp32

G = torch.bmm(vc_u.transpose(1, 2), zc_u) / chunk_len

if self.g_rel_cap is not None:
    G = project_fro_rel(
        G,
        base_norm=self.base_out_proj_weight.float().norm().detach(),
        rho=self.g_rel_cap,
    )

if decay is not None:
    deltaW = decay * deltaW + (1.0 - decay) * eta * G
else:
    deltaW = eta * G if self.no_memory_mode else deltaW + eta * G

if self.deltaW_rel_cap is not None:
    deltaW = project_fro_rel(
        deltaW,
        base_norm=self.base_out_proj_weight.float().norm().detach(),
        rho=self.deltaW_rel_cap,
    )
```

#### 4.1.3 Add new config fields and module parameters

Add to `TTTWrapper.__init__`:

- `normalize_update: bool = True`
- `norm_eps: float = 1e-6`
- `deltaW_rel_cap: float = 0.10`
- `g_rel_cap: float = 0.02`
- `decay_min: float = 0.90`
- `decay_max: float = 0.995`

When computing `decay`, use:

```python
raw = torch.sigmoid(self.log_decay_logit)
decay = self.decay_min + (self.decay_max - self.decay_min) * raw
```

Do not leave the decay unconstrained on `[0, 1]`.

#### 4.1.4 Add diagnostics

Log the following per TTT layer during training and evaluation:

- `deltaW_fro_mean`
- `deltaW_rel_mean = ||DeltaW||_F / ||W0||_F`
- `G_fro_mean`
- `G_rel_mean = ||G||_F / ||W0||_F`
- `decay`
- `effective_window_chunks = 1 / (1 - decay)`
- `effective_window_tokens = chunk_size / (1 - decay)`

These are required.

### Optional structural edit

If mandatory patch still fails, add a config flag:

- `use_residual_fast_path: bool = False`

and change apply to:

```python
base_oc = torch.bmm(zc.float(), W0.unsqueeze(0).transpose(1, 2))
fast_oc = torch.bmm(zc.float(), deltaW.transpose(1, 2))
oc = base_oc + self.fast_gate * fast_oc
if bias is not None:
    oc = oc + bias.float().unsqueeze(0).unsqueeze(0)
```

where `self.fast_gate` is a learned scalar initialized small.

## 4.2 `models/ssm_ttt_model.py`

### Mandatory edits

#### 4.2.1 Remove unconditional `.detach()` on the source

Current code:

```python
source_embeddings = hidden_states.detach()
```

Replace with a config-controlled source policy.

Add new config args:

- `ttt_target_source: "embedding" | "layer_input"`
- `ttt_detach_source: bool`

For the first patched rerun use:

- `ttt_target_source = "embedding"`
- `ttt_detach_source = false`

Implementation rule:

- if source is `"embedding"`, use embedding output;
- if source is `"layer_input"`, pass the current layer input to that TTT block;
- only detach if `ttt_detach_source == true`.

#### 4.2.2 Thread layer-local source if requested

When iterating through layers, for a `TTTMamba2Block`, compute:

- `source = source_embeddings` if mode is `embedding`
- `source = hidden_states` *before* entering the TTT layer if mode is `layer_input`

and pass that source into the wrapper.

## 4.3 `data/prepare_data.py`

### Mandatory edits

Save document offsets for the training set, not just the validation set.

Create:

- `train_offsets.npy`

Each training document should have:

- a start offset,
- an end offset,
- EOS appended exactly once between documents.

This is required for correct TTT boundary handling.

## 4.4 `data/dataloader.py`

### Mandatory edits

#### 4.4.1 Stop using the current EOS heuristic as the primary boundary mechanism

The current logic:

- samples a random global start,
- checks only the first EOS in the sampled span,
- then tries to cut after it.

This is not enough.

Replace with a doc-offset-based sampler:

1. sample a document id,
2. sample a start position **within that document**,
3. return a segment that stays within that document,
4. pad only at the document end if needed.

#### 4.4.2 Fix the worker RNG bug

Do not rely on a persistent `self.rng` inside `__getitem__` with no worker-specific reseeding.

Use either:

- deterministic indexing based on `idx`, or
- `worker_init_fn` plus a worker-local RNG derived from `(base_seed, worker_id, idx)`.

Recommended simple rule:

```python
worker_info = torch.utils.data.get_worker_info()
worker_id = 0 if worker_info is None else worker_info.id
rng = np.random.default_rng(self.seed + 1000003 * worker_id + idx)
```

Then use that `rng` to sample `doc_id` and `start`.

#### 4.4.3 Guarantee one-document samples in the pilot

For the next patched pilot, each sequence must contain exactly one document segment. Do not pack multiple documents. Do not rely on EOS-based in-sequence resets for the first rerun.

## 4.5 `evaluate.py`

### Mandatory edits

#### 4.5.1 Regenerate a longer-document validation cache

Current evaluation used too few long documents.

Regenerate validation data with:

- `min_doc_len = 32768`
- target `>= 50` documents if available
- if the public validation split is too small, draw a held-out long-document subset from train and save it separately

#### 4.5.2 Add optional eval-time diagnostics

For TTT models, during evaluation also record:

- per-layer `deltaW_rel_mean`
- per-layer `deltaW_rel_max`

for each context length.

This will immediately tell whether the 32k failure is still a state-scale problem.

---

## 5. Default patched config for the next cheap rerun

Create a new config, e.g.

`configs/tiny_pilot_ssm_ttt_v3.yaml`

with these defaults:

```yaml
model_type: ssm_ttt

model_args:
  d_model: 768
  n_layer: 24
  d_intermediate: 0
  ssm_cfg:
    layer: Mamba2
    d_state: 128
    d_conv: 4
    expand: 2
    headdim: 64
  rms_norm: true
  residual_in_fp32: true
  fused_add_norm: false

  num_ttt_layers: 4
  ttt_chunk_size: 64
  ttt_kernel_size: 5

  ttt_target_source: embedding
  ttt_detach_source: false

  ttt_inner_lr_init: 0.10
  ttt_decay_factor_init: 0.95
  ttt_decay_min: 0.90
  ttt_decay_max: 0.995

  ttt_normalize_update: true
  ttt_norm_eps: 1.0e-6
  ttt_deltaW_rel_cap: 0.10
  ttt_G_rel_cap: 0.02

seq_len: 2048
batch_size: 16
total_tokens: 200000000
lr: 0.0006
weight_decay: 0.1
betas: [0.9, 0.95]
warmup_steps: 512
grad_clip: 1.0
use_amp: true
seed: 42

data_dir: /path/to/data_cache_v2
num_workers: 4

log_interval: 50
save_interval: 5000
eval_interval: 2000
output_dir: /path/to/runs/tiny_pilot_ssm_ttt_v3
```

Important:

- the first rerun should be **cheap**;
- do **not** jump directly to 1B or 5B tokens with the current broken math.

---

## 6. Required experiment sequence

## 6.1 Stage 0 — correctness patch and tests

Before training anything new, add and pass the following tests.

### Test 0.1 — zero-update identity

If `mix_coeffs = 0`, then TTT output must equal vanilla Mamba output to numerical tolerance.

### Test 0.2 — update-state cap test

Construct synthetic `G` with large norm, run one update, and verify

\[
\|\Delta W\|_F \le \rho \|W_0\|_F.
\]

### Test 0.3 — normalized EMA sanity test

With constant `G`, verify empirically that the corrected rule

\[
\Delta W_{n+1} = \lambda \Delta W_n + (1-\lambda)\eta G
\]

saturates to roughly `eta * G`, not `eta * G / (1 - lambda)`.

### Test 0.4 — single-document sample test

Verify that every training sample lies inside one doc span from `train_offsets.npy`.

### Test 0.5 — worker randomness test

With `num_workers > 1`, confirm that the first 100 sampled start positions are not duplicated across workers due to identical RNG state.

### Test 0.6 — chunk causality test

For the target builder, confirm that `q_{t+j}` is zeroed whenever `t+j` crosses a chunk boundary.

## 6.2 Stage 1 — cheap screening runs

Use the 131M model size already in the repo.

Train each configuration for `200M` tokens and evaluate at `{2k, 4k, 8k, 16k, 32k}`.

Run exactly these 6 configs:

1. `C0`: current repo v2 (control)
2. `C1`: corrected update + relative cap + no-detach, chunk 128
3. `C2`: same as C1 but chunk 64
4. `C3`: same as C1 but chunk 32
5. `C4`: same as C2 but `ttt_target_source = layer_input`
6. `C5`: same as C2 but only late layers enabled: `[14, 19]`

### Stage-1 selection rule

A config advances if:

- `PPL_32k < 2.0 * PPL_32k(vanilla_ssm)` **and**
- no layer has `deltaW_rel_mean > 0.20` at 32k **and**
- the curve does not blow up by more than `3x` between 8k and 32k.

Keep the best **two** configs.

## 6.3 Stage 2 — stable pilot

Using the top 2 configs from Stage 1:

- train at `seq_len = 8192`
- use `1B` tokens
- keep model size at 131M, or scale modestly to ~300M if resources are available

Evaluate again on the same Figure-2 protocol.

### Stage-2 success criterion

Advance to Stage 3 only if at least one config satisfies:

- `PPL_32k <= 1.25 * PPL_32k(vanilla_ssm)`
- `PPL_16k` and `PPL_32k` are both finite and not diverging sharply
- `deltaW_rel_mean <= 0.15` on all TTT layers at 32k

If **no** config satisfies this, implement the optional structural patch from Section 3.7 and rerun Stage 1 quickly before any scaling.

## 6.4 Stage 3 — main figure run

Only after Stage 2 succeeds:

- scale to `300M–500M`
- train with `seq_len = 32768`
- token budget `>= 5B` for a serious pilot, `20B` for a full paper-style run

Train and compare:

1. vanilla SSM
2. SWA Transformer
3. patched SSM + TTT

Evaluate exactly on:

- `2k`
- `4k`
- `8k`
- `16k`
- `32k`

using a fixed scored suffix of `2048` tokens.

---

## 7. Expected outcomes and interpretation rules

## 7.1 What would count as a good result

For the patched 131M/300M pilot:

- 4k should remain competitive or improved;
- 8k should remain competitive;
- 16k and 32k should become **stable**, even if not yet better than vanilla SSM.

The immediate goal is **stability first, win second**.

## 7.2 What would count as evidence the math fix worked

The fix worked if:

- learned decays no longer cluster at the max with runaway `deltaW_rel`;
- `deltaW_rel` stays bounded while 32k PPL stays sane;
- the 4k/8k gains survive.

## 7.3 What would count as a deeper architectural limitation

Only conclude “full out-proj TTT may be too invasive for pure SSM” if all of the following are true:

- corrected update rule is used;
- cumulative fast-weight norm is capped relative to \(W_0\);
- data pipeline is fixed;
- training uses at least `seq_len = 8192`;
- smaller chunk sizes have been tried;
- late-only layers have been tried.

If it still fails after all that, then the next move is not to abandon TTT, but to switch to:

- residual fast path, or
- hybrid SSM + attention.

---

## 8. Recommended fallback if pure SSM still underperforms

If the pure SSM remains inferior after Stage 2 with the corrected math, use the same TTT machinery on a **hybrid backbone**:

- Samba-like or SWA+SSM hybrid,
- where attention handles local token mixing,
- and TTT provides long-range associative memory.

This is the most likely high-confidence fallback.

---

## 9. Things the agent must *not* do

- Do **not** spend another big run on the current `DeltaW = decay * DeltaW + G` rule.
- Do **not** assume that clipping `G` alone is sufficient.
- Do **not** rely on the current EOS heuristic and call the samples “single document”.
- Do **not** make final claims from only 5 long validation documents.
- Do **not** conclude the idea is dead before the corrected state update is tested.

---

## 10. One-sentence summary for the next coding pass

**Keep the current causal chunkwise TTT structure, but replace the fast-weight state dynamics with a normalized discounted update, cap the cumulative state relative to the base weight, fix the single-document data pipeline, and rerun a cheap 131M screening sweep before any larger experiment.**
