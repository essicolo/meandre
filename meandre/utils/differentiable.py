"""Differentiable approximations of discontinuous functions.

HYDROTEL uses hard thresholds (rain/snow at 0C, runoff when theta > theta_sat,
etc.) that create discontinuities breaking gradient flow. Every hard threshold
in the physics pipeline is replaced with a smooth approximation from this module.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def soft_threshold(x: Tensor, threshold: float, sharpness: float = 10.0) -> Tensor:
    """Differentiable approximation of (x > threshold).

    Returns ~0 when x << threshold, ~1 when x >> threshold.
    Higher sharpness makes the transition sharper (approaches a step function).
    """
    return torch.sigmoid(sharpness * (x - threshold))


def soft_clamp(x: Tensor, lo: float, hi: float, sharpness: float = 10.0) -> Tensor:
    """Differentiable clamp to [lo, hi].

    Approaches hard clamp as sharpness -> inf.
    """
    return lo + (hi - lo) * torch.sigmoid(sharpness * (x - lo)) * torch.sigmoid(
        sharpness * (hi - x)
    )


def soft_relu(x: Tensor, sharpness: float = 10.0) -> Tensor:
    """Differentiable ReLU: softplus with controllable sharpness.

    Approaches ReLU as sharpness -> inf.
    """
    return F.softplus(x * sharpness) / sharpness


def soft_step(x: Tensor, sharpness: float = 10.0) -> Tensor:
    """Differentiable step: sigmoid(sharpness * x), centered at 0."""
    return torch.sigmoid(sharpness * x)


def smooth_partition(
    value: Tensor, threshold: float, sharpness: float = 10.0
) -> tuple[Tensor, Tensor]:
    """Split value into (below_threshold, above_threshold) parts smoothly.

    Useful for rain/snow partitioning at 0C:
        rain, snow = smooth_partition(precip, T, threshold=0.0)

    Returns:
        below: value when x < threshold (e.g. snowfall fraction * precip)
        above: value when x > threshold (e.g. rainfall fraction * precip)
    """
    rain_frac = soft_threshold(threshold, threshold=0.0, sharpness=sharpness)
    snow_frac = 1.0 - rain_frac
    return snow_frac * value, rain_frac * value
