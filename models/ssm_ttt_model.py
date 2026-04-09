"""
SSM + In-Place TTT Language Model — v3.

Builds on mamba_ssm's MixerModel/MambaLMHeadModel but replaces selected
blocks with TTT-wrapped versions.

Key design:
- TTT layers are selected per spec section 4.2: 4 evenly spaced layers
- The forward pass threads source_embeddings through TTT blocks
- Non-TTT blocks use the standard Mamba2 forward
- Configurable target source: "embedding" or "layer_input"
- Configurable source detachment
"""

import math
import copy
from functools import partial
from collections import namedtuple

import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .ttt_wrapper import TTTMamba2Block


def select_ttt_layers(n_layers: int, num_ttt_layers: int = 4):
    """
    Select TTT-enabled layer indices per spec section 4.2:
    round(L/5), round(2L/5), round(3L/5), round(4L/5) for 4 layers.
    Generalized for arbitrary num_ttt_layers.
    """
    if num_ttt_layers == 0:
        return []
    indices = []
    for i in range(1, num_ttt_layers + 1):
        idx = round(i * n_layers / (num_ttt_layers + 1))
        idx = max(0, min(idx, n_layers - 1))
        indices.append(idx)
    seen = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    ssm_cfg = copy.deepcopy(ssm_cfg)
    ssm_layer = ssm_cfg.pop("layer", "Mamba2")
    if ssm_layer not in ["Mamba1", "Mamba2"]:
        raise ValueError(f"Invalid ssm_layer: {ssm_layer}")
    mixer_cls = partial(
        Mamba2 if ssm_layer == "Mamba2" else Mamba,
        layer_idx=layer_idx,
        **ssm_cfg,
        **factory_kwargs
    )

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class SSMTTTModel(nn.Module):
    """
    SSM backbone with optional TTT on selected layers.

    When ttt_layer_indices is empty or None, this is a vanilla SSM.
    When ttt_layer_indices is provided, those layers get TTT wrapping.
    """

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = True,
        # TTT params
        ttt_layer_indices=None,
        ttt_chunk_size: int = 256,
        ttt_kernel_size: int = 5,
        ttt_clip_tau: float = None,
        ttt_inner_lr_init: float = 0.10,
        ttt_decay_factor_init: float = 0.95,
        ttt_decay_min: float = 0.90,
        ttt_decay_max: float = 0.995,
        ttt_normalize_update: bool = True,
        ttt_norm_eps: float = 1e-6,
        ttt_deltaW_rel_cap: float = 0.10,
        ttt_G_rel_cap: float = 0.02,
        ttt_use_residual_fast_path: bool = False,
        ttt_target_source: str = "embedding",
        ttt_detach_source: bool = False,
        pad_vocab_size_multiple: int = 8,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.n_layer = n_layer
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.ttt_layer_indices = set(ttt_layer_indices or [])
        self.ttt_target_source = ttt_target_source
        self.ttt_detach_source = ttt_detach_source

        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        layers = []
        for i in range(n_layer):
            block = create_block(
                d_model,
                d_intermediate=d_intermediate,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                **factory_kwargs,
            )

            if i in self.ttt_layer_indices:
                ttt_block = TTTMamba2Block(
                    original_block=block,
                    d_model=d_model,
                    chunk_size=ttt_chunk_size,
                    target_kernel_size=ttt_kernel_size,
                    clip_tau=ttt_clip_tau,
                    inner_lr_init=ttt_inner_lr_init,
                    decay_factor_init=ttt_decay_factor_init,
                    decay_min=ttt_decay_min,
                    decay_max=ttt_decay_max,
                    normalize_update=ttt_normalize_update,
                    norm_eps=ttt_norm_eps,
                    deltaW_rel_cap=ttt_deltaW_rel_cap,
                    g_rel_cap=ttt_G_rel_cap,
                    use_residual_fast_path=ttt_use_residual_fast_path,
                    device=device,
                    dtype=dtype,
                )
                layers.append(ttt_block)
            else:
                layers.append(block)

        self.layers = nn.ModuleList(layers)

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        self.lm_head.weight = self.embedding.weight

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,
            )
        )

    def forward(self, input_ids, labels=None, segment_ids=None, **kwargs):
        """
        Args:
            input_ids: [B, T] token ids
            labels: [B, T] labels for loss computation (shifted internally)
            segment_ids: [B, T] document segment ids for TTT boundary handling (optional)

        Returns:
            CausalLMOutput with logits and optional loss
        """
        hidden_states = self.embedding(input_ids)

        if self.ttt_target_source == "embedding":
            source_embeddings = hidden_states
            if self.ttt_detach_source:
                source_embeddings = source_embeddings.detach()
        else:
            source_embeddings = None

        residual = None
        for layer in self.layers:
            if isinstance(layer, TTTMamba2Block):
                if self.ttt_target_source == "layer_input":
                    if not self.fused_add_norm:
                        layer_source = (hidden_states + residual) if residual is not None else hidden_states
                    else:
                        layer_source = hidden_states
                    if self.ttt_detach_source:
                        layer_source = layer_source.detach()
                    src = layer_source
                else:
                    src = source_embeddings

                hidden_states, residual = layer(
                    hidden_states, residual,
                    source_embeddings=src,
                    segment_ids=segment_ids,
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual,
                )

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        CausalLMOutput = namedtuple("CausalLMOutput", ["loss", "logits"])
        return CausalLMOutput(loss=loss, logits=logits)

    def get_ttt_diagnostics(self):
        """Collect TTT health diagnostics from all TTT layers."""
        diagnostics = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TTTMamba2Block):
                wrapper = layer.ttt_wrapper
                diag = wrapper._last_diagnostics
                for key, val in diag.items():
                    diagnostics[f"layer_{i}/{key}"] = val
        return diagnostics

    def count_parameters(self):
        """Count total and per-component parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        ttt_params = 0
        for layer in self.layers:
            if isinstance(layer, TTTMamba2Block):
                ttt_params += sum(
                    p.numel() for p in layer.ttt_wrapper.target_builder.parameters()
                )
                ttt_params += 1  # log_inner_lr
                if layer.ttt_wrapper.log_decay_logit is not None:
                    ttt_params += 1  # log_decay_logit
                if layer.ttt_wrapper.use_residual_fast_path:
                    ttt_params += 1  # fast_gate
        return {
            "total": total,
            "ttt_extra": ttt_params,
            "backbone": total - ttt_params,
        }


def create_vanilla_ssm(
    d_model: int = 768,
    n_layer: int = 24,
    d_intermediate: int = 0,
    vocab_size: int = 50280,
    ssm_cfg=None,
    **kwargs,
):
    """Create a vanilla SSM (no TTT) for baseline."""
    if ssm_cfg is None:
        ssm_cfg = {"layer": "Mamba2", "d_state": 128, "d_conv": 4, "expand": 2, "headdim": 64}
    # Filter out TTT-specific kwargs that don't apply to vanilla SSM
    ttt_keys = [k for k in kwargs if k.startswith('ttt_')]
    for k in ttt_keys:
        kwargs.pop(k)
    return SSMTTTModel(
        d_model=d_model,
        n_layer=n_layer,
        d_intermediate=d_intermediate,
        vocab_size=vocab_size,
        ssm_cfg=ssm_cfg,
        ttt_layer_indices=[],
        **kwargs,
    )


def create_ssm_ttt(
    d_model: int = 768,
    n_layer: int = 24,
    d_intermediate: int = 0,
    vocab_size: int = 50280,
    ssm_cfg=None,
    num_ttt_layers: int = 4,
    ttt_layer_indices_override=None,
    ttt_chunk_size: int = 256,
    ttt_kernel_size: int = 5,
    ttt_clip_tau: float = None,
    ttt_inner_lr_init: float = 0.10,
    ttt_decay_factor_init: float = 0.95,
    ttt_decay_min: float = 0.90,
    ttt_decay_max: float = 0.995,
    ttt_normalize_update: bool = True,
    ttt_norm_eps: float = 1e-6,
    ttt_deltaW_rel_cap: float = 0.10,
    ttt_G_rel_cap: float = 0.02,
    ttt_use_residual_fast_path: bool = False,
    ttt_target_source: str = "embedding",
    ttt_detach_source: bool = False,
    **kwargs,
):
    """Create an SSM with in-place TTT."""
    if ssm_cfg is None:
        ssm_cfg = {"layer": "Mamba2", "d_state": 128, "d_conv": 4, "expand": 2, "headdim": 64}

    if ttt_layer_indices_override is not None:
        ttt_layers = list(ttt_layer_indices_override)
    else:
        ttt_layers = select_ttt_layers(n_layer, num_ttt_layers)
    print(f"TTT-enabled layer indices: {ttt_layers}")

    return SSMTTTModel(
        d_model=d_model,
        n_layer=n_layer,
        d_intermediate=d_intermediate,
        vocab_size=vocab_size,
        ssm_cfg=ssm_cfg,
        ttt_layer_indices=ttt_layers,
        ttt_chunk_size=ttt_chunk_size,
        ttt_kernel_size=ttt_kernel_size,
        ttt_clip_tau=ttt_clip_tau,
        ttt_inner_lr_init=ttt_inner_lr_init,
        ttt_decay_factor_init=ttt_decay_factor_init,
        ttt_decay_min=ttt_decay_min,
        ttt_decay_max=ttt_decay_max,
        ttt_normalize_update=ttt_normalize_update,
        ttt_norm_eps=ttt_norm_eps,
        ttt_deltaW_rel_cap=ttt_deltaW_rel_cap,
        ttt_G_rel_cap=ttt_G_rel_cap,
        ttt_use_residual_fast_path=ttt_use_residual_fast_path,
        ttt_target_source=ttt_target_source,
        ttt_detach_source=ttt_detach_source,
        **kwargs,
    )
