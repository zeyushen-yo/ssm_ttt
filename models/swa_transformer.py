"""
Sliding-Window Attention (SWA) Transformer baseline.

A standard decoder-only Transformer with sliding-window (local) attention,
matching the paper's reference configuration for fair comparison.

Uses Flash Attention 2 with sliding window support for efficiency.
"""

import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)."""

    def __init__(self, dim, max_seq_len=65536, base=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_len):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(x.dtype)
            self._sin_cached = emb.sin().to(x.dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]

    def forward(self, q, k):
        seq_len = q.shape[2]
        cos, sin = self._update_cos_sin_tables(q, seq_len)
        return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin):
    # x: [B, H, T, D], cos/sin: [T, D]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class SWATransformerBlock(nn.Module):
    """Decoder block with sliding-window causal attention + FFN."""

    def __init__(self, d_model, num_heads, d_ff, window_size, dropout=0.0,
                 device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size

        self.norm1 = nn.RMSNorm(d_model, eps=1e-5, **factory_kwargs)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-5, **factory_kwargs)

        self.q_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.o_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)

        self.gate_proj = nn.Linear(d_model, d_ff, bias=False, **factory_kwargs)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False, **factory_kwargs)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False, **factory_kwargs)

        self.rotary = RotaryEmbedding(self.head_dim, device=device)

    def forward(self, x, residual=None):
        B, T, D = x.shape

        if residual is not None:
            x = x + residual
        residual = x

        # Pre-norm + Attention
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)

        # Try Flash Attention with sliding window
        attn_out = self._attention(q, k, v, T)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        h = self.o_proj(attn_out)

        # Residual
        x = h + residual
        residual = x

        # Pre-norm + FFN (SwiGLU)
        h = self.norm2(x)
        h = F.silu(self.gate_proj(h)) * self.up_proj(h)
        h = self.down_proj(h)

        return h, residual

    def _attention(self, q, k, v, T):
        """Sliding-window causal attention using Flash Attention 2 when available."""
        try:
            from flash_attn import flash_attn_func
            # flash_attn_func expects [B, T, H, D]
            q_fa = q.transpose(1, 2)  # [B, T, H, D]
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            out = flash_attn_func(
                q_fa, k_fa, v_fa,
                causal=True,
                window_size=(self.window_size - 1, 0),  # (left, right) - causal so right=0
            )
            return out.transpose(1, 2)  # back to [B, H, T, D]
        except (ImportError, Exception):
            pass

        # Fallback: manual sliding-window causal attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=q.device, dtype=q.dtype),
            diagonal=1
        )
        # Sliding window mask: mask out positions more than window_size away
        window_mask = torch.tril(
            torch.full((T, T), float('-inf'), device=q.device, dtype=q.dtype),
            diagonal=-(self.window_size)
        )
        mask = causal_mask + window_mask
        attn_weights = attn_weights + mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)


class SWATransformerLM(nn.Module):
    """
    Sliding-Window Attention Transformer Language Model.

    Spec section 6, Phase 3 reference config:
    - d_model=1024, n_layer=24, d_ff=3072, num_heads=8
    - window_size=2048, seq_len=32768
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_layer: int = 24,
        d_ff: int = 3072,
        num_heads: int = 8,
        vocab_size: int = 50280,
        window_size: int = 2048,
        dropout: float = 0.0,
        pad_vocab_size_multiple: int = 8,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.layers = nn.ModuleList([
            SWATransformerBlock(
                d_model, num_heads, d_ff, window_size,
                dropout=dropout, **factory_kwargs
            )
            for _ in range(n_layer)
        ])
        self.norm_f = nn.RMSNorm(d_model, eps=1e-5, **factory_kwargs)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        # Rescale output projections
        for i, layer in enumerate(self.layers):
            with torch.no_grad():
                layer.o_proj.weight /= math.sqrt(2 * self.n_layer)
                layer.down_proj.weight /= math.sqrt(2 * self.n_layer)

    def forward(self, input_ids, labels=None, **kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        # Final residual + norm
        if residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = self.norm_f(hidden_states)

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

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total": total}
