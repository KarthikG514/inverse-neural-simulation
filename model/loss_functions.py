"""
Loss functions and parameter normalization utilities
for Inverse Neural Simulation.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

# Parameter ranges used across the project
PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "gravity": (5.0, 15.0),
    "mass": (0.5, 5.0),
    "friction": (0.0, 1.0),
    "restitution": (0.3, 1.0),
}

PARAM_NAMES = ["gravity", "mass", "friction", "restitution"]


class ParameterNormalizer:
    """
    Utility to normalize/denormalize physics parameters.

    Normalization:
        norm = (x - min) / (max - min)
    Denormalization:
        x = norm * (max - min) + min
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.min_vals = torch.tensor(
            [PARAM_RANGES[p][0] for p in PARAM_NAMES], device=self.device
        )
        self.max_vals = torch.tensor(
            [PARAM_RANGES[p][1] for p in PARAM_NAMES], device=self.device
        )
        self.range_vals = self.max_vals - self.min_vals

    def normalize(self, params: torch.Tensor) -> torch.Tensor:
        """
        Normalize parameters to [0, 1].

        params: (B, 4) in original scale.
        returns: (B, 4) in [0, 1]
        """
        return (params - self.min_vals) / self.range_vals

    def denormalize(self, params_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize parameters back to original scale.

        params_norm: (B, 4) in [0, 1]
        returns: (B, 4) in original units
        """
        return params_norm * self.range_vals + self.min_vals


class ParameterRegressionLoss(nn.Module):
    """
    L2 loss on normalized parameters.

    IMPORTANT:
    - DataLoader must already provide normalized targets in [0, 1].
    - Model outputs are trained to approximate these normalized targets.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self, pred_params: torch.Tensor, true_params_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_params: (B, 4) model outputs (should approximate [0, 1])
            true_params_norm: (B, 4) ground truth already normalized to [0, 1]

        Returns:
            scalar MSE loss
        """
        return self.mse(pred_params, true_params_norm)


def _test_loss():
    """Simple sanity test for loss + normalizer."""
    device = "cpu"
    normalizer = ParameterNormalizer(device=device)
    loss_fn = ParameterRegressionLoss(device=device)

    # Create dummy raw params in original scale
    raw = torch.tensor(
        [
            [5.0, 0.5, 0.0, 0.3],   # mins
            [10.0, 2.75, 0.5, 0.65],  # mids
            [15.0, 5.0, 1.0, 1.0],  # maxs
        ],
        dtype=torch.float32,
        device=device,
    )

    norm = normalizer.normalize(raw)
    rec = normalizer.denormalize(norm)
    assert torch.allclose(raw, rec, atol=1e-6), "Denormalization mismatch"

    # Perfect prediction -> zero loss
    loss_perfect = loss_fn(norm, norm)
    print("Perfect prediction loss:", loss_perfect.item())

    # Bad prediction -> non-zero loss
    zeros = torch.zeros_like(norm)
    loss_bad = loss_fn(zeros, norm)
    print("Bad prediction loss:", loss_bad.item())


if __name__ == "__main__":
    _test_loss()
