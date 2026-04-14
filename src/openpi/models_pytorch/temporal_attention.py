"""MEM (Multi-Scale Embodied Memory) temporal attention module.

Per the MEM paper (arxiv 2603.03596):
  "The video encoder does not introduce new learnable parameters compared to
   standard ViTs. Video encoding capabilities are added by modifying the
   attention pattern of the ViT and adding fixed sinusoidal temporal position
   encoding, allowing initialization of weights from pre-trained ViT weights."
  "The temporal attention performs attention over timesteps' representations
   for the same image patch using a causal attention mask."

Accordingly this module provides:
  - `TemporalPositionEncoding`: fixed sinusoidal temporal position encoding (no learnable params).
  - `apply_temporal_attention`: a stateless function that runs factorized temporal
    self-attention by REUSING the hosting transformer layer's existing Q/K/V/O
    projection weights, and applies a causal mask across frames.
"""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TemporalPositionEncoding(nn.Module):
    """Sinusoidal temporal position encoding for multi-frame inputs.

    Generates position embeddings based on ``frame_index * frame_interval`` (seconds),
    using sine-cosine encoding with logarithmically spaced frequencies. This module
    has no learnable parameters.
    """

    def __init__(self, d_model: int, max_frames: int = 16, min_period: float = 0.01, max_period: float = 10.0):
        super().__init__()
        self.d_model = d_model
        self.max_frames = max_frames
        self.min_period = min_period
        self.max_period = max_period

    def forward(self, num_frames: int, frame_interval: float, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Compute temporal position encoding.

        Returns:
            Tensor of shape ``[1, N, 1, d_model]`` broadcastable over batch and spatial dims.
        """
        if self.d_model % 2 != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by 2")

        frame_times = torch.arange(num_frames, device=device, dtype=torch.float64) * frame_interval

        fraction = torch.linspace(0.0, 1.0, self.d_model // 2, dtype=torch.float64, device=device)
        period = self.min_period * (self.max_period / self.min_period) ** fraction

        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = scaling_factor[None, :] * frame_times[:, None]  # [N, d_model//2]
        encoding = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)  # [N, d_model]

        return encoding.to(dtype=dtype).unsqueeze(0).unsqueeze(2)  # [1, N, 1, d_model]


def apply_temporal_attention(
    hidden_states: Tensor,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    o_proj: nn.Linear,
    temporal_pos_enc: TemporalPositionEncoding,
    num_image_tokens_per_frame: int,
    num_frames: int,
    frame_interval: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tensor:
    """Factorized temporal attention that reuses the hosting layer's projections.

    Applies causal self-attention across the temporal dimension for image tokens,
    with each spatial position attended independently. Non-image tokens pass through
    unchanged. Introduces NO new learnable parameters.

    Args:
        hidden_states: ``[B, seq_len, D]`` prefix embeddings (image tokens first, frame-major).
        q_proj, k_proj, v_proj, o_proj: the hosting PaliGemma layer's existing projections.
        temporal_pos_enc: shared (non-learnable) sinusoidal position encoding module.
        num_image_tokens_per_frame: S (e.g. 768 = 3 cameras * 256 patches).
        num_frames: N (total frames; current + past).
        frame_interval: time interval between frames in seconds.
        num_heads: number of query heads (for Q projection reshape).
        num_kv_heads: number of key/value heads (for GQA; K/V reshape).
        head_dim: per-head dim.

    Returns:
        ``[B, seq_len, D]``: hidden states with temporal attention applied to image tokens.
    """
    B, seq_len, D = hidden_states.shape
    S = num_image_tokens_per_frame
    N = num_frames
    num_img_total = N * S

    img_tokens = hidden_states[:, :num_img_total, :]  # [B, N*S, D], frame-major
    other_tokens = hidden_states[:, num_img_total:, :]  # [B, rest, D]
    img_residual = img_tokens

    # [B, N*S, D] -> [B, N, S, D]; add temporal pos enc; -> [B*S, N, D]
    img_tokens = img_tokens.view(B, N, S, D)
    pos_enc = temporal_pos_enc(N, frame_interval, hidden_states.device, hidden_states.dtype)
    img_tokens = img_tokens + pos_enc  # broadcasts over B and S
    img_tokens = img_tokens.permute(0, 2, 1, 3).reshape(B * S, N, D)

    # Reuse the hosting layer's projections (NO new learnable weights).
    proj_dtype = q_proj.weight.dtype
    img_tokens_proj = img_tokens.to(proj_dtype)

    q = q_proj(img_tokens_proj).view(B * S, N, num_heads, head_dim).transpose(1, 2)  # [B*S, Hq, N, hd]
    k = k_proj(img_tokens_proj).view(B * S, N, num_kv_heads, head_dim).transpose(1, 2)  # [B*S, Hkv, N, hd]
    v = v_proj(img_tokens_proj).view(B * S, N, num_kv_heads, head_dim).transpose(1, 2)  # [B*S, Hkv, N, hd]

    # GQA: broadcast KV heads to match Q heads.
    if num_kv_heads != num_heads:
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        repeats = num_heads // num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    # Scaled dot-product with CAUSAL mask across the temporal dimension (paper).
    scaling = head_dim**-0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling  # [B*S, Hq, N, N]
    causal_mask = torch.triu(
        torch.ones(N, N, device=hidden_states.device, dtype=torch.bool), diagonal=1
    )
    attn_weights = attn_weights.masked_fill(causal_mask[None, None, :, :], float("-inf"))
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)  # [B*S, Hq, N, hd]

    # Reuse layer's O projection.
    attn_output = attn_output.transpose(1, 2).reshape(B * S, N, num_heads * head_dim)
    attn_output = o_proj(attn_output)  # [B*S, N, D]

    # [B*S, N, D] -> [B, S, N, D] -> [B, N, S, D] -> [B, N*S, D]
    attn_output = attn_output.view(B, S, N, D).permute(0, 2, 1, 3).reshape(B, num_img_total, D)
    attn_output = attn_output.to(img_residual.dtype)

    img_tokens_out = img_residual + attn_output
    return torch.cat([img_tokens_out, other_tokens], dim=1)
