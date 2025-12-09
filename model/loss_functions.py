"""
Loss Functions for Physics Parameter Prediction.

Losses:
1. ParameterNormalizer: Utility to normalize/denormalize parameters
2. ParameterRegressionLoss: L2 loss on normalized parameters (PRIMARY)
3. TrajectoryMatchingLoss: Trajectory matching loss (SECONDARY, Week 3+)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

# Parameter ranges (MUST match dataset generation)
PARAM_RANGES = {
    'gravity': (5.0, 15.0),
    'mass': (0.5, 5.0),
    'friction': (0.0, 1.0),
    'restitution': (0.3, 1.0),
}

PARAM_NAMES = ['gravity', 'mass', 'friction', 'restitution']

class ParameterNormalizer:
    """
    Utility to normalize/denormalize physics parameters.
    
    Why needed: Gravity range (5-15) is much larger than friction range (0-1).
    Normalization puts all parameters on equal footing for training.
    """
    
    def __init__(self, param_ranges: Dict[str, Tuple[float, float]] = PARAM_RANGES):
        """
        Args:
            param_ranges: Dict of parameter names to (min, max) tuples
        """
        self.param_ranges = param_ranges
        self.param_names = list(param_ranges.keys())
    
    def normalize(self, params: torch.Tensor) -> torch.Tensor:
        """
        Normalize parameters to [0, 1].
        
        Args:
            params: (B, 4) raw parameters in original scale
        
        Returns:
            normalized: (B, 4) in [0, 1]
        
        TODO: For each parameter, apply min-max scaling
        """
        pass
    
    def denormalize(self, params_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize parameters back to original scale.
        
        Args:
            params_norm: (B, 4) normalized to [0, 1]
        
        Returns:
            params: (B, 4) in original scale
        
        TODO: Reverse the normalization
        """
        pass

class ParameterRegressionLoss(nn.Module):
    """
    L2 loss on normalized parameters.
    
    This is the PRIMARY loss for Weeks 1-2.
    """
    
    def __init__(self, param_ranges: Dict = PARAM_RANGES):
        super().__init__()
        self.normalizer = ParameterNormalizer(param_ranges)
    
    def forward(self, pred_params: torch.Tensor, 
                true_params: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 loss on normalized parameters.
        
        Args:
            pred_params: (B, 4) raw predictions (unnormalized)
            true_params: (B, 4) ground truth (original scale)
        
        Returns:
            loss: scalar
        
        TODO:
        - Normalize both predictions and ground truth
        - Clamp predictions to [0, 1]
        - Compute MSE
        - Return scalar loss
        """
        pass

class TrajectoryMatchingLoss(nn.Module):
    """
    Trajectory matching loss (SECONDARY, Week 3+).
    
    Requires a differentiable physics simulator.
    Skeleton provided; will be filled in Week 3.
    """
    
    def __init__(self, simulator=None):
        super().__init__()
        self.simulator = simulator
    
    def forward(self, pred_params: torch.Tensor,
                true_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compare simulated trajectory to ground truth.
        
        Args:
            pred_params: (B, 4) predicted parameters
            true_trajectory: (B, 60, 3) ground truth trajectory
        
        Returns:
            loss: scalar
        
        TODO (Week 3):
        - Simulate using pred_params
        - Compute L2 distance
        - Average over batch and time
        """
        if self.simulator is None:
            raise ValueError("Simulator not provided for trajectory loss")
        
        # Placeholder: will implement in Week 3
        return torch.tensor(0.0, device=pred_params.device)

def test_losses():
    """Test loss functions with dummy data."""
    print("Testing ParameterNormalizer...")
    normalizer = ParameterNormalizer()
    
    # Dummy params in original scale
    dummy_params = torch.tensor([
        [9.81, 1.5, 0.3, 0.8],
        [10.0, 2.0, 0.2, 0.9],
    ])
    
    print(f"Original params: {dummy_params}")
    # TODO: Test normalize/denormalize
    # normalized = normalizer.normalize(dummy_params)
    # print(f"Normalized: {normalized}")
    
    print("\nTesting ParameterRegressionLoss...")
    loss_fn = ParameterRegressionLoss()
    
    pred_params = torch.randn(2, 4)
    true_params = dummy_params
    
    # TODO: Compute loss
    # loss = loss_fn(pred_params, true_params)
    # print(f"Loss: {loss.item():.4f}")

if __name__ == "__main__":
    test_losses()
