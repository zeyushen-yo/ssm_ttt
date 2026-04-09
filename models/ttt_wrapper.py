"""
TTT wrapper for SSM blocks — v4 with error-corrective delta rule.

Implements the updated spec v4:
- Three update rules: hebb (v3 baseline), delta_current, delta_base
- Three scale modes: mean, sqrt_len, sum
- Normalized discounted update: DeltaW = decay * DeltaW + (1-decay) * eta * G
- RMS normalization on update inputs
- Relative Frobenius cap on both G and cumulative DeltaW
- Bounded decay range [decay_min, decay_max]
- Optional disable_updates for TTT-on/off evaluation control
- Optional surprise-based write gating
- Optional centered updates (mean subtraction)
- Optional residual fast path (backup only)

The execution order per chunk:
1. Compute frozen pre-output features Z_c
2. Apply current W_0 + DeltaW_c
3. Compute target Vhat_c
4. Compute update G_c (hebb, delta_current, or delta_base)
5. Update DeltaW_{c+1}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .target_builder import TargetBuilder


def rms_norm_lastdim(x, eps=1e-6):
    """Tokenwise RMS normalization along the last dimension."""
    rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return x / rms


def project_fro_rel(deltaW, base_norm, rho, eps=1e-8):
    """
    Project deltaW to Frobenius ball of radius rho * base_norm.
    deltaW: [B, d_out, d_in]
    """
    dw_norm = torch.norm(deltaW.reshape(deltaW.shape[0], -1), dim=1, keepdim=True).unsqueeze(-1)
    max_norm = rho * base_norm
    scale = torch.clamp(max_norm / (dw_norm + eps), max=1.0)
    return deltaW * scale


class TTTWrapper(nn.Module):
    """
    Wraps a Mamba2 block to add in-place TTT on its out_proj.

    The wrapper:
    - Runs the Mamba2 SSM forward to get pre-output features (everything before out_proj)
    - Applies (W_0 + DeltaW) chunkwise with apply-then-update
    - The DeltaW is per batch item and accumulates across chunks
    """

    def __init__(
        self,
        mamba_block,
        d_model: int,
        chunk_size: int = 256,
        target_kernel_size: int = 5,
        clip_tau: float = None,
        inner_lr_init: float = 0.10,
        decay_factor_init: float = 0.95,
        decay_min: float = 0.90,
        decay_max: float = 0.995,
        normalize_update: bool = True,
        norm_eps: float = 1e-6,
        deltaW_rel_cap: float = 0.10,
        g_rel_cap: float = 0.02,
        use_residual_fast_path: bool = False,
        update_rule: str = "hebb",
        scale_mode: str = "mean",
        normalize_z: bool = True,
        normalize_err: bool = False,
        disable_updates: bool = False,
        write_gate: str = "none",
        write_gate_max: float = 3.0,
        center_updates: bool = False,
        center_beta: float = 0.95,
        target_builder: "TargetBuilder" = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.mamba_block = mamba_block
        self.d_model = d_model
        self.d_inner = mamba_block.d_inner
        self.chunk_size = chunk_size
        self.clip_tau = clip_tau
        self.normalize_update = normalize_update
        self.norm_eps = norm_eps
        self.deltaW_rel_cap = deltaW_rel_cap
        self.g_rel_cap = g_rel_cap
        self.decay_min = decay_min
        self.decay_max = decay_max
        self.use_residual_fast_path = use_residual_fast_path
        self.use_normalized_ema = normalize_update

        self.update_rule = update_rule
        self.scale_mode = scale_mode
        self.normalize_z = normalize_z
        self.normalize_err = normalize_err
        self.disable_updates = disable_updates
        self.write_gate = write_gate
        self.write_gate_max = write_gate_max
        self.center_updates = center_updates
        self.center_beta = center_beta

        assert update_rule in ("hebb", "delta_current", "delta_base"), \
            f"Unknown update_rule: {update_rule}"
        assert scale_mode in ("mean", "sqrt_len", "sum"), \
            f"Unknown scale_mode: {scale_mode}"
        assert write_gate in ("none", "chunk_err"), \
            f"Unknown write_gate: {write_gate}"

        if target_builder is not None:
            self.target_builder = target_builder
        else:
            self.target_builder = TargetBuilder(
                d_model=d_model,
                kernel_size=target_kernel_size,
                device=device,
                dtype=torch.float32,
            )

        self.log_inner_lr = nn.Parameter(
            torch.tensor(inner_lr_init, device=device, dtype=torch.float32).log()
        )

        if decay_factor_init is not None:
            raw_init = (decay_factor_init - decay_min) / (decay_max - decay_min)
            raw_init = max(1e-4, min(raw_init, 1.0 - 1e-4))
            self.log_decay_logit = nn.Parameter(
                torch.tensor(raw_init, device=device, dtype=torch.float32).logit()
            )
        else:
            self.log_decay_logit = None

        if use_residual_fast_path:
            self.fast_gate = nn.Parameter(
                torch.tensor(0.01, device=device, dtype=torch.float32)
            )

        self.base_out_proj_weight = mamba_block.out_proj.weight
        self.base_out_proj_bias = mamba_block.out_proj.bias

        self._last_diagnostics = {}
        self._running_G_mean = None

    def _get_decay(self):
        """Compute bounded decay in [decay_min, decay_max] (spec eq. 6)."""
        if self.log_decay_logit is None:
            return None
        raw = torch.sigmoid(self.log_decay_logit)
        return self.decay_min + (self.decay_max - self.decay_min) * raw

    def _get_pre_out_features(self, u, seq_idx=None):
        """
        Run Mamba2 forward path up to (but not including) out_proj.
        Returns pre-output features y of shape [B, T, d_inner].
        """
        block = self.mamba_block

        batch, seqlen, dim = u.shape

        zxbcdt = block.in_proj(u)

        A = -torch.exp(block.A_log.float())
        dt_limit_kwargs = {} if block.dt_limit == (0.0, float("inf")) else dict(dt_limit=block.dt_limit)

        d_mlp = (zxbcdt.shape[-1] - 2 * block.d_ssm - 2 * block.ngroups * block.d_state - block.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, block.d_ssm, block.d_ssm + 2 * block.ngroups * block.d_state, block.nheads],
            dim=-1
        )

        try:
            from causal_conv1d import causal_conv1d_fn
        except ImportError:
            causal_conv1d_fn = None

        if causal_conv1d_fn is None or block.activation not in ["silu", "swish"]:
            xBC = block.act(
                block.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(block.d_conv - 1)]
            )
        else:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2),
                rearrange(block.conv1d.weight, "d 1 w -> d w"),
                bias=block.conv1d.bias,
                activation=block.activation,
                seq_idx=seq_idx,
            ).transpose(1, 2)

        x, B_mat, C_mat = torch.split(
            xBC,
            [block.d_ssm, block.ngroups * block.d_state, block.ngroups * block.d_state],
            dim=-1
        )

        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=block.headdim),
            dt,
            A,
            rearrange(B_mat, "b l (g n) -> b l g n", g=block.ngroups),
            rearrange(C_mat, "b l (g n) -> b l g n", g=block.ngroups),
            chunk_size=block.chunk_size,
            D=rearrange(block.D, "(h p) -> h p", p=block.headdim) if block.D_has_hdim else block.D,
            z=rearrange(z, "b l (h p) -> b l h p", p=block.headdim) if not block.rmsnorm else None,
            dt_bias=block.dt_bias,
            dt_softplus=True,
            seq_idx=seq_idx,
            **dt_limit_kwargs,
        )

        y = rearrange(y, "b l h p -> b l (h p)")

        if block.rmsnorm:
            y = block.norm(y, z)

        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        return y  # [B, T, d_inner]

    def _apply_subspan(self, z_sub, deltaW, W0, W0_t, bias):
        """Apply current DeltaW to a subspan of features. Returns output o_sub."""
        if self.use_residual_fast_path:
            base_o = torch.bmm(z_sub.float(), W0_t)
            fast_o = torch.bmm(z_sub.float(), deltaW.transpose(1, 2))
            o = base_o + self.fast_gate * fast_o
        else:
            W_eff = W0.unsqueeze(0) + deltaW
            o = torch.bmm(z_sub.float(), W_eff.transpose(1, 2))
        if bias is not None:
            o = o + bias.float().unsqueeze(0).unsqueeze(0)
        return o

    def _compute_update_matrix(self, z_sub, v_sub, o_sub, W0, W0_t):
        """
        Compute the gradient matrix G based on the configured update_rule.

        For hebb:         G = V_hat^T @ Z_norm  (original)
        For delta_current: G = E^T @ Z_norm  where E = V_hat - sg(O_current)
        For delta_base:    G = E^T @ Z_norm  where E = V_hat - sg(Z @ W0^T)
        """
        B = z_sub.shape[0]
        span_len = z_sub.shape[1]
        zc_fp32 = z_sub.float()
        vc_fp32 = v_sub.float()

        if self.normalize_z:
            zc_u = rms_norm_lastdim(zc_fp32, eps=self.norm_eps)
        else:
            zc_u = zc_fp32

        if self.update_rule == "hebb":
            if self.normalize_update:
                vc_u = rms_norm_lastdim(vc_fp32, eps=self.norm_eps)
            else:
                vc_u = vc_fp32
            signal = vc_u
        elif self.update_rule == "delta_current":
            o_current = o_sub.detach().float()
            error = vc_fp32 - o_current
            if self.normalize_err:
                error = rms_norm_lastdim(error, eps=self.norm_eps)
            signal = error
        elif self.update_rule == "delta_base":
            o_base = torch.bmm(zc_fp32, W0_t).detach()
            error = vc_fp32 - o_base
            if self.normalize_err:
                error = rms_norm_lastdim(error, eps=self.norm_eps)
            signal = error
        else:
            raise ValueError(f"Unknown update_rule: {self.update_rule}")

        G = torch.bmm(signal.transpose(1, 2), zc_u)

        if self.scale_mode == "mean":
            G = G / span_len
        elif self.scale_mode == "sqrt_len":
            G = G / (span_len ** 0.5)

        err_rms = None
        if self.update_rule in ("delta_current", "delta_base"):
            err_rms = signal.square().mean(dim=-1).sqrt().mean().item()

        return G, err_rms

    def _apply_update(self, G, deltaW, W0_norm, eta, decay, write_gate_val=None):
        """Apply the update G to deltaW with decay, gating, centering, and caps."""
        B = G.shape[0]

        if self.clip_tau is not None:
            G_norm = torch.norm(G.reshape(B, -1), dim=1, keepdim=True).unsqueeze(-1)
            scale = torch.clamp(self.clip_tau / (G_norm + 1e-8), max=1.0)
            G = G * scale

        if self.g_rel_cap is not None:
            G = project_fro_rel(G, base_norm=W0_norm, rho=self.g_rel_cap)

        G_fro = torch.norm(G.reshape(B, -1), dim=1).mean().item()

        if self.center_updates:
            if self._running_G_mean is None:
                self._running_G_mean = G.detach().clone()
            else:
                self._running_G_mean = self.center_beta * self._running_G_mean + \
                    (1.0 - self.center_beta) * G.detach()
            G = G - self._running_G_mean

        if write_gate_val is not None:
            G = write_gate_val * G

        if decay is not None:
            if self.use_normalized_ema:
                deltaW = decay * deltaW + (1.0 - decay) * eta * G
            else:
                deltaW = decay * deltaW + eta * G
        else:
            deltaW = deltaW + eta * G

        if self.deltaW_rel_cap is not None:
            deltaW = project_fro_rel(deltaW, base_norm=W0_norm, rho=self.deltaW_rel_cap)

        return deltaW, G_fro

    def forward(self, u, source_embeddings, segment_ids=None, seq_idx=None):
        """
        TTT-enabled forward pass with boundary-aware resets.

        Args:
            u: input to the Mamba2 block [B, T, d_model]
            source_embeddings: token embeddings for target building [B, T, d_model]
            segment_ids: [B, T] document segment ids for boundary-aware reset (optional)
            seq_idx: optional sequence index for causal conv

        Returns:
            out: block output [B, T, d_model] (to be added to residual stream)
            deltaW: final fast weight state [B, d_model, d_inner]
        """
        B, T, d = u.shape
        C = self.chunk_size

        if segment_ids is not None and B > 1:
            raise ValueError(
                "Boundary-aware TTT with segment_ids is only supported for batch_size=1. "
                f"Got B={B}. Either set boundary_aware=false or use batch_size=1."
            )

        z = self._get_pre_out_features(u, seq_idx=seq_idx)  # [B, T, d_inner]
        vhat = self.target_builder(source_embeddings, chunk_size=C, segment_ids=segment_ids)  # [B, T, d_model]

        W0 = self.base_out_proj_weight.float()  # [d_model, d_inner]
        W0_t = W0.unsqueeze(0).transpose(1, 2)  # [1, d_inner, d_model]
        bias = self.base_out_proj_bias
        W0_norm = W0.norm().detach()

        deltaW = torch.zeros(B, self.d_model, self.d_inner, device=u.device, dtype=torch.float32)

        eta = self.log_inner_lr.exp()
        decay = self._get_decay()

        G_fro_accum = 0.0
        G_rel_accum = 0.0
        err_rms_accum = 0.0
        write_gate_accum = 0.0
        n_updates = 0
        n_resets = 0
        prev_seg_id = None
        n_subspans_total = 0
        n_chunks_with_boundary = 0
        n_chunks_total = 0

        outputs = []
        for s in range(0, T, C):
            e = min(s + C, T)

            zc = z[:, s:e, :]
            vc = vhat[:, s:e, :]

            n_chunks_total += 1

            if segment_ids is not None:
                seg_chunk = segment_ids[0, s:e]
                chunk_len = e - s

                subspan_starts = [0]
                for i in range(1, chunk_len):
                    if seg_chunk[i].item() != seg_chunk[i - 1].item():
                        subspan_starts.append(i)
                subspan_starts.append(chunk_len)

                n_subspans_in_chunk = len(subspan_starts) - 1
                n_subspans_total += n_subspans_in_chunk
                if n_subspans_in_chunk > 1:
                    n_chunks_with_boundary += 1

                chunk_outputs = []
                for sp_idx in range(len(subspan_starts) - 1):
                    sp_s = subspan_starts[sp_idx]
                    sp_e = subspan_starts[sp_idx + 1]
                    cur_seg_id = seg_chunk[sp_s].item()

                    if prev_seg_id is not None and cur_seg_id != prev_seg_id:
                        deltaW = torch.zeros_like(deltaW)
                        n_resets += 1

                    z_sub = zc[:, sp_s:sp_e, :]
                    v_sub = vc[:, sp_s:sp_e, :]

                    o_sub = self._apply_subspan(z_sub, deltaW, W0, W0_t, bias)
                    chunk_outputs.append(o_sub.to(u.dtype))

                    if sp_e - sp_s > 0 and not self.disable_updates:
                        G, err_rms = self._compute_update_matrix(
                            z_sub, v_sub, o_sub, W0, W0_t)

                        write_gate_val = None
                        if self.write_gate == "chunk_err" and err_rms is not None:
                            write_gate_val = min(err_rms, self.write_gate_max)
                            write_gate_accum += write_gate_val

                        deltaW, g_fro = self._apply_update(
                            G, deltaW, W0_norm, eta, decay, write_gate_val)
                        G_fro_accum += g_fro
                        G_rel_accum += g_fro / (W0_norm.item() + 1e-8)
                        if err_rms is not None:
                            err_rms_accum += err_rms
                        n_updates += 1

                    prev_seg_id = cur_seg_id

                oc = torch.cat(chunk_outputs, dim=1)
                outputs.append(oc)
            else:
                oc = self._apply_subspan(zc, deltaW, W0, W0_t, bias)
                outputs.append(oc.to(u.dtype))

                if e - s > 0 and not self.disable_updates:
                    G, err_rms = self._compute_update_matrix(
                        zc, vc, oc, W0, W0_t)

                    write_gate_val = None
                    if self.write_gate == "chunk_err" and err_rms is not None:
                        write_gate_val = min(err_rms, self.write_gate_max)
                        write_gate_accum += write_gate_val

                    deltaW, g_fro = self._apply_update(
                        G, deltaW, W0_norm, eta, decay, write_gate_val)
                    G_fro_accum += g_fro
                    G_rel_accum += g_fro / (W0_norm.item() + 1e-8)
                    if err_rms is not None:
                        err_rms_accum += err_rms
                    n_updates += 1

        out = torch.cat(outputs, dim=1)

        dW_fro = torch.norm(deltaW.reshape(B, -1), dim=1).mean().item()
        diag = {
            "deltaW_fro_mean": dW_fro,
            "deltaW_rel_mean": dW_fro / (W0_norm.item() + 1e-8),
            "G_fro_mean": G_fro_accum / max(n_updates, 1),
            "G_rel_mean": G_rel_accum / max(n_updates, 1),
            "decay": decay.item() if decay is not None else 0.0,
            "effective_window_chunks": 1.0 / (1.0 - decay.item() + 1e-8) if decay is not None else float('inf'),
            "effective_window_tokens": C / (1.0 - decay.item() + 1e-8) if decay is not None else float('inf'),
            "inner_lr": eta.item(),
            "n_resets": n_resets,
            "update_rule": self.update_rule,
        }
        if n_updates > 0 and self.update_rule in ("delta_current", "delta_base"):
            diag["err_rms_mean"] = err_rms_accum / max(n_updates, 1)
        if self.write_gate == "chunk_err" and n_updates > 0:
            diag["write_gate_mean"] = write_gate_accum / max(n_updates, 1)
        if self.use_residual_fast_path:
            diag["fast_gate"] = self.fast_gate.item()
        if n_chunks_total > 0 and segment_ids is not None:
            diag["n_doc_boundaries"] = n_resets
            diag["frac_chunks_with_boundary"] = n_chunks_with_boundary / max(n_chunks_total, 1)
            diag["avg_subspans_per_chunk"] = n_subspans_total / max(n_chunks_total, 1)
        self._last_diagnostics = diag

        return out, deltaW


class TTTMamba2Block(nn.Module):
    """
    A complete Block replacement that includes the TTT wrapper.
    Mirrors the interface of mamba_ssm.modules.block.Block but with TTT.
    """

    def __init__(
        self,
        original_block,
        d_model: int,
        chunk_size: int = 256,
        target_kernel_size: int = 5,
        clip_tau: float = None,
        inner_lr_init: float = 0.10,
        decay_factor_init: float = 0.95,
        decay_min: float = 0.90,
        decay_max: float = 0.995,
        normalize_update: bool = True,
        norm_eps: float = 1e-6,
        deltaW_rel_cap: float = 0.10,
        g_rel_cap: float = 0.02,
        use_residual_fast_path: bool = False,
        update_rule: str = "hebb",
        scale_mode: str = "mean",
        normalize_z: bool = True,
        normalize_err: bool = False,
        disable_updates: bool = False,
        write_gate: str = "none",
        write_gate_max: float = 3.0,
        center_updates: bool = False,
        center_beta: float = 0.95,
        target_builder: "TargetBuilder" = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.residual_in_fp32 = original_block.residual_in_fp32
        self.fused_add_norm = original_block.fused_add_norm
        self.norm = original_block.norm
        self.mlp = original_block.mlp if hasattr(original_block, 'mlp') else None
        if self.mlp is not None and hasattr(original_block, 'norm2'):
            self.norm2 = original_block.norm2

        self.ttt_wrapper = TTTWrapper(
            mamba_block=original_block.mixer,
            d_model=d_model,
            chunk_size=chunk_size,
            target_kernel_size=target_kernel_size,
            clip_tau=clip_tau,
            inner_lr_init=inner_lr_init,
            decay_factor_init=decay_factor_init,
            decay_min=decay_min,
            decay_max=decay_max,
            normalize_update=normalize_update,
            norm_eps=norm_eps,
            deltaW_rel_cap=deltaW_rel_cap,
            g_rel_cap=g_rel_cap,
            use_residual_fast_path=use_residual_fast_path,
            update_rule=update_rule,
            scale_mode=scale_mode,
            normalize_z=normalize_z,
            normalize_err=normalize_err,
            disable_updates=disable_updates,
            write_gate=write_gate,
            write_gate_max=write_gate_max,
            center_updates=center_updates,
            center_beta=center_beta,
            target_builder=target_builder,
            device=device,
            dtype=dtype,
        )

        self.layer_idx = getattr(original_block, 'layer_idx', None)
        self._last_deltaW = None

    def forward(self, hidden_states, residual=None, inference_params=None,
                source_embeddings=None, segment_ids=None, **mixer_kwargs):
        """Forward with TTT. Requires source_embeddings to be passed through."""
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, RMSNorm
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )

        assert source_embeddings is not None, "TTT blocks require source_embeddings"
        hidden_states, deltaW = self.ttt_wrapper(
            hidden_states, source_embeddings,
            segment_ids=segment_ids,
            seq_idx=mixer_kwargs.get('seq_idx', None),
        )

        self._last_deltaW = deltaW.detach()

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, RMSNorm
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.ttt_wrapper.mamba_block.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )
