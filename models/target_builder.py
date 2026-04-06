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

    def forward(self, q: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """
        Build targets for all positions.

        Args:
            q: source sequence [B, T, d] (token embeddings for from-scratch track)
            chunk_size: chunk size C for boundary masking

        Returns:
            vhat: [B, T, d] target directions
        """
        B, T, d = q.shape
        q_fp32 = q.float()

        # Build shifted versions with within-chunk boundary zeroing
        # For each offset j in 1..K, create q_{t+j} with zeros at chunk boundaries
        u = torch.zeros(B, T, d, device=q.device, dtype=torch.float32)

        for j in range(1, self.kernel_size + 1):
            if j >= T:
                break

            # Shift q left by j positions: q_shifted[t] = q[t+j]
            q_shifted = torch.zeros_like(q_fp32)
            q_shifted[:, :T - j, :] = q_fp32[:, j:, :]

            # Zero out positions where t+j crosses a chunk boundary
            # Position t is in chunk t // C. Position t+j is in chunk (t+j) // C.
            # If they differ, zero it out.
            pos = torch.arange(T, device=q.device)
            chunk_of_t = pos // chunk_size
            chunk_of_tj = torch.clamp(pos + j, max=T - 1) // chunk_size
            # Also zero where t+j >= T
            cross_boundary = (chunk_of_t != chunk_of_tj) | (pos + j >= T)
            # cross_boundary: [T] bool mask
            q_shifted[:, cross_boundary, :] = 0.0

            # Elementwise multiply with d_j and accumulate
            u = u + self.mix_coeffs[j - 1].unsqueeze(0).unsqueeze(0) * q_shifted

        # Apply projection: vhat = u @ W_tgt^T -> [B, T, d]
        vhat = torch.matmul(u, self.W_tgt.t())

        return vhat
