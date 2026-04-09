"""
LM-aligned target builder for in-place TTT.

Implements Section 4.6 of the project spec:
  u_t^l = sum_{j=1}^{K} (d_j^l ⊙ q_{t+j})
  vhat_t^l = u_t^l @ W_tgt^l

With boundary rule: q_{t+j} = 0 if t+j crosses chunk or document boundary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetBuilder(nn.Module):
    """Builds future-looking LM-aligned targets for one TTT-enabled layer."""

    def __init__(self, d_model: int, kernel_size: int = 5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        # Depthwise future-mixing coefficients d_1..d_K, initialized to zero (spec 5.4)
        self.mix_coeffs = nn.Parameter(
            torch.zeros(kernel_size, d_model, device=device, dtype=torch.float32)
        )

        # Trainable projection W_tgt in R^(d x d), initialized diagonal (spec 5.4)
        w_tgt = torch.zeros(d_model, d_model, device=device, dtype=torch.float32)
        init_std = 0.02
        diag_vals = torch.empty(d_model, device=device, dtype=torch.float32).uniform_(-init_std, init_std)
        w_tgt.diagonal().copy_(diag_vals)
        self.W_tgt = nn.Parameter(w_tgt)

    def forward(self, q: torch.Tensor, chunk_size: int, segment_ids=None) -> torch.Tensor:
        """
        Build targets for all positions.

        Args:
            q: source sequence [B, T, d] (token embeddings for from-scratch track)
            chunk_size: chunk size C for boundary masking
            segment_ids: [B, T] document segment ids for boundary-aware masking (optional)

        Returns:
            vhat: [B, T, d] target directions
        """
        B, T, d = q.shape
        q_fp32 = q.float()

        u = torch.zeros(B, T, d, device=q.device, dtype=torch.float32)

        pos = torch.arange(T, device=q.device)
        chunk_of_t = pos // chunk_size

        for j in range(1, self.kernel_size + 1):
            if j >= T:
                break

            q_shifted = torch.zeros_like(q_fp32)
            q_shifted[:, :T - j, :] = q_fp32[:, j:, :]

            chunk_of_tj = torch.clamp(pos + j, max=T - 1) // chunk_size
            cross_chunk = (chunk_of_t != chunk_of_tj)
            cross_end = (pos + j >= T)
            cross_boundary = cross_chunk | cross_end  # [T]

            if segment_ids is not None:
                seg_t = segment_ids[:, :]  # [B, T]
                seg_tj = torch.zeros_like(seg_t)
                seg_tj[:, :T - j] = segment_ids[:, j:]
                seg_tj[:, T - j:] = -1
                cross_doc = (seg_t != seg_tj)  # [B, T]
                cross_boundary_full = cross_boundary.unsqueeze(0) | cross_doc  # [B, T]
                q_shifted = q_shifted.masked_fill(cross_boundary_full.unsqueeze(-1), 0.0)
            else:
                q_shifted[:, cross_boundary, :] = 0.0

            u = u + self.mix_coeffs[j - 1].unsqueeze(0).unsqueeze(0) * q_shifted

        vhat = torch.matmul(u, self.W_tgt.t())

        return vhat
