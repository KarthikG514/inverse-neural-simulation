"""
Neural Network Architecture for Physics Parameter Prediction.

This module contains the core architecture:
- VideoEncoder: 3D CNN to extract spatial-temporal features from video
- TemporalAggregator: Aggregate temporal information (pooling)
- PhysicsRegressor: MLP to predict 4 physics parameters
- InversePhysicsNet: Complete end-to-end model
"""

import torch
import torch.nn as nn
from typing import Tuple

class VideoEncoder(nn.Module):
    """3D CNN to extract features from video."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(video))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.mean(x, dim=(2, 3, 4))
        return x

class TemporalAggregator(nn.Module):
    """Aggregate temporal features using FC layer."""
    
    def __init__(self, in_channels: int = 128, out_dim: int = 512):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(features))

class PhysicsRegressor(nn.Module):
    """MLP to predict 4 physics parameters."""
    
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.fc3 = nn.Linear(128, 4)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class InversePhysicsNet(nn.Module):
    """Complete model: VideoEncoder -> TemporalAggregator -> PhysicsRegressor."""
    
    def __init__(self):
        super().__init__()
        self.encoder = VideoEncoder()
        self.aggregator = TemporalAggregator(in_channels=128, out_dim=512)
        self.regressor = PhysicsRegressor(in_dim=512, hidden_dim=256)
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        features = self.encoder(video)
        aggregated = self.aggregator(features)
        params = self.regressor(aggregated)
        return params

def test_model():
    """Test model with dummy data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InversePhysicsNet().to(device)
    
    dummy_video = torch.randn(2, 1, 60, 64, 64).to(device)
    
    print(f"Input shape: {dummy_video.shape}")
    output = model(dummy_video)
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (2, 4), "Output shape mismatch!"
    print("✓ Model test passed!")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

if __name__ == "__main__":
    test_model()
