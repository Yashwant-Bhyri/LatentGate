"""Wan 2.1 trajectory-guided RoPE primitives for the LatentGate pilot.

This module isolates the box-to-token and STD-RoPE operations used by DiTraj
from its version-specific Diffusers pipeline patch. It is not a complete video
pipeline integration. The published method is described in
https://arxiv.org/abs/2509.21839 and the official implementation is
https://github.com/xduzhangjiayu/DiTraj (CC BY-NC 4.0).

The optional phase strength is a LatentGate extension: 0 preserves the original
RoPE, 1 reproduces hard STD-RoPE alignment, and values in between interpolate
on the unit circle so rotary magnitudes remain unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _bounds(mask: torch.Tensor) -> tuple[int, int, int, int]:
    """Return a filled rectangular mask's half-open (y1, y2, x1, x2) bounds."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D frame mask, got shape {tuple(mask.shape)}")
    ys, xs = torch.where(mask.bool())
    if ys.numel() == 0:
        raise ValueError("Each latent frame must contain a non-empty foreground box")
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    expected = (y2 - y1) * (x2 - x1)
    if int(mask.bool().sum()) != expected:
        raise ValueError("Foreground mask must be one filled axis-aligned rectangle per frame")
    return y1, y2, x1, x2


def _nearest_indices(old_size: int, new_size: int, device: torch.device) -> torch.Tensor:
    if old_size < 1 or new_size < 1:
        raise ValueError("RoI dimensions must be positive")
    return torch.linspace(0, old_size - 1, new_size, device=device).round().long()


def _nearest_resize_grid(grid: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Nearest-resize an [H,W,...] grid with the rounding used by DiTraj."""
    yi = _nearest_indices(grid.shape[0], out_h, grid.device)
    xi = _nearest_indices(grid.shape[1], out_w, grid.device)
    return grid[yi[:, None], xi[None, :]]


def boxes_to_latent_mask(
    trajectory_yxyx: Sequence[Sequence[float]],
    *,
    num_video_frames: int,
    token_height: int,
    token_width: int,
    temporal_stride: int = 4,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Rasterize normalized [frame,y1,y2,x1,x2] boxes at Wan token resolution.

    Wan's temporal VAE maps 4n+1 video frames to n+1 latent frames. The box path
    is interpolated in video-frame coordinates and sampled at [0,4,8,...], as
    in the released DiTraj runner.
    """
    if num_video_frames < 1 or (num_video_frames - 1) % temporal_stride != 0:
        raise ValueError("Wan frame count must be 4n+1 for temporal_stride=4")
    if token_height < 1 or token_width < 1:
        raise ValueError("Token grid dimensions must be positive")
    anchors = sorted((list(row) for row in trajectory_yxyx), key=lambda row: row[0])
    if not anchors:
        raise ValueError("At least one trajectory anchor is required")
    if any(len(row) != 5 for row in anchors):
        raise ValueError("Each anchor must be [frame,y1,y2,x1,x2]")
    if anchors[0][0] < 0 or anchors[-1][0] >= num_video_frames:
        raise ValueError("Trajectory anchor is outside the video frame range")

    latent_frames = (num_video_frames - 1) // temporal_stride + 1
    mask = torch.zeros(
        (latent_frames, token_height, token_width), dtype=torch.bool, device=device
    )

    for latent_idx in range(latent_frames):
        video_idx = latent_idx * temporal_stride
        if video_idx <= anchors[0][0]:
            vals = anchors[0][1:]
        elif video_idx >= anchors[-1][0]:
            vals = anchors[-1][1:]
        else:
            left = max(i for i, row in enumerate(anchors) if row[0] <= video_idx)
            a, b = anchors[left], anchors[left + 1]
            alpha = (video_idx - a[0]) / (b[0] - a[0])
            vals = [a[j] + alpha * (b[j] - a[j]) for j in range(1, 5)]

        y1n, y2n, x1n, x2n = map(float, vals)
        if not (0 <= y1n < y2n <= 1 and 0 <= x1n < x2n <= 1):
            raise ValueError(f"Invalid normalized box at frame {video_idx}: {vals}")
        y1 = min(token_height - 1, int(y1n * token_height))
        y2 = min(token_height, max(y1 + 1, int(y2n * token_height)))
        x1 = min(token_width - 1, int(x1n * token_width))
        x2 = min(token_width, max(x1 + 1, int(x2n * token_width)))
        mask[latent_idx, y1:y2, x1:x2] = True

    return mask


def apply_std_rope(
    rotary_emb: torch.Tensor,
    foreground_mask: torch.Tensor,
    *,
    strength: float | Sequence[float] | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Align foreground spatial RoPE to the smallest-box anchor in each frame.

    Args:
        rotary_emb: Complex RoPE values shaped [latent_frames, token_h, token_w, channels].
        foreground_mask: One filled rectangular object box per latent frame, [T,H,W].
        strength: Per-run scalar or one [0,1] alignment blend per latent frame.

    As in the official code, the spatial source patch is read from each target
    frame's own RoPE map. This aligns spatial positions while retaining that
    frame's temporal phase. The phase blend interpolates along the shorter arc
    on the unit circle rather than linearly shrinking RoPE magnitudes.
    """
    if rotary_emb.ndim != 4 or not rotary_emb.is_complex():
        raise ValueError("rotary_emb must be a complex [T,H,W,C] tensor")
    if foreground_mask.ndim != 3:
        raise ValueError("foreground_mask must have shape [T,H,W]")
    if tuple(rotary_emb.shape[:3]) != tuple(foreground_mask.shape):
        raise ValueError(
            f"RoPE grid {tuple(rotary_emb.shape[:3])} and box mask "
            f"{tuple(foreground_mask.shape)} differ"
        )

    boxes = [_bounds(foreground_mask[t]) for t in range(foreground_mask.shape[0])]
    areas = [(y2 - y1) * (x2 - x1) for y1, y2, x1, x2 in boxes]
    anchor_idx = min(range(len(areas)), key=areas.__getitem__)
    ay1, ay2, ax1, ax2 = boxes[anchor_idx]

    if isinstance(strength, torch.Tensor):
        strengths = strength.detach().flatten().tolist()
    elif isinstance(strength, Sequence) and not isinstance(strength, (str, bytes)):
        strengths = list(strength)
    else:
        strengths = [float(strength)] * foreground_mask.shape[0]
    if len(strengths) != foreground_mask.shape[0]:
        raise ValueError("strength must be scalar or contain one value per latent frame")
    if any(not 0.0 <= float(value) <= 1.0 for value in strengths):
        raise ValueError("RoPE alignment strength values must be in [0,1]")

    result = rotary_emb.clone()
    for t, (y1, y2, x1, x2) in enumerate(boxes):
        target_h, target_w = y2 - y1, x2 - x1
        source = rotary_emb[t, ay1:ay2, ax1:ax2]
        canonical = _nearest_resize_grid(source, target_h, target_w)
        current = rotary_emb[t, y1:y2, x1:x2]
        alpha = float(strengths[t])
        if alpha == 0.0:
            continue
        phase_delta = torch.angle(canonical * current.conj())
        phase_rotation = torch.polar(torch.ones_like(phase_delta), phase_delta * alpha)
        result[t, y1:y2, x1:x2] = current * phase_rotation
    return result


def build_rtoken_attention_mask(
    foreground_mask: torch.Tensor,
    *,
    alignment_strength: float | Sequence[float] | torch.Tensor = 1.0,
    max_tokens: int = 8192,
) -> torch.Tensor:
    """Build DiTraj's keep-mask, excluding redundant background↔foreground links.

    The dense result is [N,N] bool for PyTorch SDPA (True means attend). A
    token-count guard prevents accidental quadratic allocations for long clips.
    Duplicate positions are counted with ``torch.unique`` instead of forming
    the much larger [N,N,3] equality tensor used in the reference helper.
    """
    if foreground_mask.ndim != 3:
        raise ValueError("foreground_mask must have shape [T,H,W]")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    t_count, height, width = foreground_mask.shape
    n_tokens = t_count * height * width
    if n_tokens > max_tokens:
        raise ValueError(
            f"Refusing dense {n_tokens}x{n_tokens} attention mask; "
            f"max_tokens={max_tokens}"
        )
    if isinstance(alignment_strength, torch.Tensor):
        strengths = alignment_strength.detach().flatten().tolist()
    elif isinstance(alignment_strength, Sequence) and not isinstance(alignment_strength, (str, bytes)):
        strengths = list(alignment_strength)
    else:
        strengths = [float(alignment_strength)] * t_count
    if len(strengths) != t_count or any(not 0.0 <= float(value) <= 1.0 for value in strengths):
        raise ValueError("alignment_strength must be in [0,1], scalar or one value per latent frame")
    if any(float(value) != 1.0 for value in strengths):
        # The repeated-token exclusion is defined for exact coordinate
        # replacement. Fractional phasor alignment has no exact integer-grid
        # collision map, so do not apply the hard-alignment mask to it.
        return torch.ones((n_tokens, n_tokens), dtype=torch.bool, device=foreground_mask.device)
    boxes = [_bounds(foreground_mask[t]) for t in range(t_count)]
    areas = [(y2 - y1) * (x2 - x1) for y1, y2, x1, x2 in boxes]
    anchor_idx = min(range(len(areas)), key=areas.__getitem__)
    ay1, ay2, ax1, ax2 = boxes[anchor_idx]

    device = foreground_mask.device
    tt, yy, xx = torch.meshgrid(
        torch.arange(t_count, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    coords = torch.stack((tt, yy, xx), dim=-1)
    for t, (y1, y2, x1, x2) in enumerate(boxes):
        target_h, target_w = y2 - y1, x2 - x1
        anchor_y = torch.arange(ay1, ay2, device=device)
        anchor_x = torch.arange(ax1, ax2, device=device)
        source_y = _nearest_indices(anchor_y.numel(), target_h, device)
        source_x = _nearest_indices(anchor_x.numel(), target_w, device)
        coords[t, y1:y2, x1:x2, 1] = anchor_y[source_y][:, None]
        coords[t, y1:y2, x1:x2, 2] = anchor_x[source_x][None, :]

    flat_coords = coords.reshape(-1, 3)
    _, inverse, counts = torch.unique(
        flat_coords, dim=0, return_inverse=True, return_counts=True
    )
    duplicated = counts[inverse] > 1
    is_foreground = foreground_mask.reshape(-1).bool()
    redundant_background = duplicated & ~is_foreground
    blocked = (
        redundant_background[:, None] & is_foreground[None, :]
    ) | (
        is_foreground[:, None] & redundant_background[None, :]
    )
    return ~blocked
