"""
Neural Network Architecture for Physics Parameter Prediction.

This module contains the core architecture:
- VideoEncoder: 3D CNN to extract spatial-temporal features from video
- TemporalAggregator: Aggregate temporal information (pooling or attention)
- PhysicsRegressor: MLP to predict 4 physics parameters
- InversePhysicsNet: Complete end-to-end model
"""

import torch
import torch.nn as nn
from typing import Tuple

class VideoEncoder(nn.Module):
    """
    3D CNN to extract features from video.
    
    Input: (B, 1, 60, 64, 64) - batch, channels, frames, height, width
    Output: (B, 128) - global feature vector
    """
    
    def __init__(self):
        super().__init__()
        # TODO: Implement 3D CNN layers
        pass
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, 1, 60, 64, 64)
        Returns:
            features: (B, 128)
        """
        # TODO: Implement forward pass
        pass

class TemporalAggregator(nn.Module):
    """
    Aggregate temporal features using pooling or attention.
    """
    
    def __init__(self, in_channels: int = 128, out_dim: int = 512):
        super().__init__()
        # TODO: Implement aggregation
        pass
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, 128) or (B, 128, T)
        Returns:
            aggregated: (B, 512)
        """
        # TODO: Implement forward pass
        pass

class PhysicsRegressor(nn.Module):
    """
    MLP to predict 4 physics parameters: gravity, mass, friction, restitution.
    
    Input: (B, 512) global feature vector
    Output: (B, 4) unnormalized parameters
    """
    
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        # TODO: Implement MLP layers
        pass
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, 512)
        Returns:
            params: (B, 4)
        """
        # TODO: Implement forward pass
        pass

class InversePhysicsNet(nn.Module):
    """
    Complete model: VideoEncoder → TemporalAggregator → PhysicsRegressor
    
    Input: Video (B, 1, 60, 64, 64)
    Output: Physics parameters (B, 4)
    """
    
    def __init__(self):
        super().__init__()
        self.encoder = VideoEncoder()
        self.aggregator = TemporalAggregator(in_channels=128, out_dim=512)
        self.regressor = PhysicsRegressor(in_dim=512, hidden_dim=256)
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through entire network.
        
        Args:
            video: (B, 1, 60, 64, 64)
        Returns:
            params: (B, 4)
        """
        # TODO: Chain forward passes
        pass

def test_model():
    """Test model with dummy data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InversePhysicsNet().to(device)
    
    # Dummy input
    dummy_video = torch.randn(2, 1, 60, 64, 64).to(device)
    
    print(f"Input shape: {dummy_video.shape}")
    # TODO: Forward pass test
    # output = model(dummy_video)
    # print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    test_model()
