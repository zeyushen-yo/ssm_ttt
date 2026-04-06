"""
TTT wrapper for SSM blocks.

Implements Sections 4.3-4.9 and 5.1-5.6 of the project spec:
- Intercepts a Mamba2 block to expose pre-output features
- Applies chunkwise apply-then-update with per-batch-item fast weights
- Uses the TargetBuilder for LM-aligned targets

The execution order per chunk (spec 5.6):
1. Compute frozen pre-output features Z_c
2. Apply current W_0 + DeltaW_c
3. Compute Vhat_c
4. Update DeltaW_{c+1}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .target_builder import TargetBuilder


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
        inner_lr_init: float = 1.0,
        decay_factor_init: float = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.mamba_block = mamba_block
        self.d_model = d_model
        self.d_inner = mamba_block.d_inner
        self.chunk_size = chunk_size
        self.clip_tau = clip_tau

        self.target_builder = TargetBuilder(
            d_model=d_model,
            kernel_size=target_kernel_size,
            device=device,
            dtype=torch.float32,
        )

        # Learnable inner learning rate (per-layer scalar in log-space for stability)
        self.log_inner_lr = nn.Parameter(
            torch.tensor(inner_lr_init, device=device, dtype=torch.float32).log()
        )

        # Optional learnable decay factor for DeltaW (EMA-style forgetting)
        if decay_factor_init is not None:
            self.log_decay_logit = nn.Parameter(
                torch.tensor(decay_factor_init, device=device, dtype=torch.float32).logit()
            )
        else:
            self.log_decay_logit = None

        # Store reference to original out_proj weight (spec 4.3: W_0^l)
        # Shape: [d_model, d_inner]
        self.base_out_proj_weight = mamba_block.out_proj.weight
        self.base_out_proj_bias = mamba_block.out_proj.bias

    def _get_pre_out_features(self, u, seq_idx=None):
        """
        Run Mamba2 forward path up to (but not including) out_proj.
        Returns pre-output features y of shape [B, T, d_inner].

        Uses the non-fused (slow) path to intercept before out_proj.
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

        # Convolution
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

    def forward(self, u, source_embeddings, seq_idx=None):
        """
        TTT-enabled forward pass.

        Args:
            u: input to the Mamba2 block [B, T, d_model]
            source_embeddings: token embeddings for target building [B, T, d_model]
            seq_idx: optional sequence index for causal conv

        Returns:
            out: block output [B, T, d_model] (to be added to residual stream)
        """
        B, T, d = u.shape
        C = self.chunk_size

        # Step 1: Get pre-output features (frozen SSM forward)
        z = self._get_pre_out_features(u, seq_idx=seq_idx)  # [B, T, d_inner]

        # Step 2: Build LM-aligned targets
        vhat = self.target_builder(source_embeddings, chunk_size=C)  # [B, T, d_model]

        # Step 3: Chunkwise apply-then-update loop
        W0 = self.base_out_proj_weight.float()  # [d_model, d_inner]
        bias = self.base_out_proj_bias

        # Per-batch-item fast weights (spec 5.2)
        deltaW = torch.zeros(B, self.d_model, self.d_inner, device=u.device, dtype=torch.float32)

        eta = self.log_inner_lr.exp()
        decay = torch.sigmoid(self.log_decay_logit) if self.log_decay_logit is not None else None

        outputs = []
        for s in range(0, T, C):
            e = min(s + C, T)
            chunk_len = e - s

            zc = z[:, s:e, :]       # [B, chunk_len, d_inner]
            vc = vhat[:, s:e, :]     # [B, chunk_len, d_model]

            # Apply: O = Z @ (W_0 + DeltaW)^T  (spec 4.7)
            W_eff = W0.unsqueeze(0) + deltaW  # [B, d_model, d_inner]
            oc = torch.bmm(zc.float(), W_eff.transpose(1, 2))  # [B, chunk_len, d_model]

            if bias is not None:
                oc = oc + bias.float().unsqueeze(0).unsqueeze(0)

            outputs.append(oc.to(u.dtype))

            # Update: G = eta * (1/C) * Vhat^T @ Z  (spec 4.9 + learnable inner LR)
            G = eta * torch.bmm(vc.float().transpose(1, 2), zc.float()) / chunk_len  # [B, d_model, d_inner]

            # Clipping (spec 4.10)
            if self.clip_tau is not None:
                G_norm = torch.norm(G.reshape(B, -1), dim=1, keepdim=True).unsqueeze(-1)
                scale = torch.clamp(self.clip_tau / (G_norm + 1e-8), max=1.0)
                G = G * scale

            if decay is not None:
                deltaW = decay * deltaW + G
            else:
                deltaW = deltaW + G

        out = torch.cat(outputs, dim=1)  # [B, T, d_model]
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
        inner_lr_init: float = 1.0,
        decay_factor_init: float = None,
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
            device=device,
            dtype=dtype,
        )

        self.layer_idx = getattr(original_block, 'layer_idx', None)
        self._last_deltaW = None
        self._last_G_norm = None

    def forward(self, hidden_states, residual=None, inference_params=None,
                source_embeddings=None, **mixer_kwargs):
        """
        Forward with TTT. Requires source_embeddings to be passed through.
        """
        # Residual + Norm (same as original Block)
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

        # TTT-wrapped Mamba2 forward
        assert source_embeddings is not None, "TTT blocks require source_embeddings"
        hidden_states, deltaW = self.ttt_wrapper(
            hidden_states, source_embeddings,
            seq_idx=mixer_kwargs.get('seq_idx', None),
        )

        # Store diagnostics
        self._last_deltaW = deltaW.detach()

        # MLP (if present, same as original Block)
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
