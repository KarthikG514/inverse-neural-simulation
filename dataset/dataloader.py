"""
PyTorch DataLoader for Inverse Physics Dataset.

Features:
- Load MP4 videos from disk
- Normalize video pixels to [0, 1]
- Normalize physics parameters to [0, 1]
- Create train/val/test splits
- Return batches with proper tensor shapes
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# Parameter ranges (MUST match loss_functions.py)
PARAM_RANGES = {
    'gravity': (5.0, 15.0),
    'mass': (0.5, 5.0),
    'friction': (0.0, 1.0),
    'restitution': (0.3, 1.0),
}

class InversePhysicsDataset(Dataset):
    """
    PyTorch Dataset for physics parameter prediction from video.
    
    Loads videos and returns normalized tensors.
    """
    
    def __init__(self, video_dir: str, metadata_path: str,
                 split: str = 'train', train_ratio: float = 0.8,
                 val_ratio: float = 0.1):
        """
        Initialize dataset.
        
        Args:
            video_dir: Path to videos folder
            metadata_path: Path to metadata.json
            split: 'train', 'val', or 'test'
            train_ratio: Fraction for training (0.8)
            val_ratio: Fraction for validation (0.1)
        
        TODO:
        - Load metadata.json
        - Split indices into train/val/test
        - Store paths and parameters
        """
        self.video_dir = Path(video_dir)
        self.split = split
        
        # TODO: Load metadata
        # with open(metadata_path) as f:
        #     self.metadata = json.load(f)
        
        print(f"[InversePhysicsDataset] Initialized {split} split")
    
    def __len__(self) -> int:
        """Return number of samples in this split."""
        # TODO: Return length of data in current split
        pass
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load one sample.
        
        Returns:
            (video_tensor, param_tensor)
            video_tensor: (1, 60, 64, 64) normalized to [0, 1]
            param_tensor: (4,) normalized to [0, 1]
        
        TODO:
        - Load video from MP4
        - Resize to 64×64
        - Normalize to [0, 1]
        - Load parameters from metadata
        - Normalize parameters to [0, 1]
        - Return as torch tensors
        """
        pass
    
    def load_video(self, video_path: str) -> np.ndarray:
        """
        Load MP4 video and return frames.
        
        Args:
            video_path: Path to MP4 file
        
        Returns:
            numpy array of shape (60, 64, 64) with values in [0, 255]
        
        TODO:
        - Open video with cv2.VideoCapture
        - Read all frames
        - Ensure exactly 60 frames
        - Convert to grayscale
        - Resize to 64×64
        - Return as numpy array
        """
        pass
    
    def normalize_video(self, video: np.ndarray) -> torch.Tensor:
        """
        Normalize video to [0, 1].
        
        Converts (60, 64, 64) → (1, 60, 64, 64), divides by 255
        
        TODO: Implement
        """
        pass
    
    def normalize_params(self, params: Dict[str, float]) -> torch.Tensor:
        """
        Normalize physics parameters to [0, 1].
        
        TODO:
        - Apply min-max scaling using PARAM_RANGES
        - Output: tensor of shape (4,) in order [gravity, mass, friction, restitution]
        """
        pass

def create_dataloaders(data_dir: str = "data/videos",
                       metadata_path: str = "data/videos/metadata.json",
                       batch_size: int = 32,
                       num_workers: int = 0) -> Dict[str, DataLoader]:
    """
    Create train, val, test dataloaders.
    
    Args:
        data_dir: Video directory
        metadata_path: Metadata JSON file
        batch_size: Batch size for training
        num_workers: Number of worker processes (0 for Windows)
    
    Returns:
        Dict with keys 'train', 'val', 'test' containing DataLoaders
    
    TODO:
    - Create InversePhysicsDataset for each split
    - Create DataLoader for each
    - Set shuffle=True for train, shuffle=False for val/test
    - Return dict of dataloaders
    """
    pass

def validate_dataloader(dataloader: DataLoader, num_batches: int = 2):
    """
    Check that dataloader returns correct shapes and values.
    
    TODO:
    - Iterate through num_batches
    - Check video shape: (B, 1, 60, 64, 64)
    - Check video values in [0, 1]
    - Check param shape: (B, 4)
    - Check param values in [0, 1]
    - Print sample info
    """
    print("[validate_dataloader] Checking batch shapes and values...")
    
    for batch_idx, (videos, params) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Videos shape: {videos.shape}")
        print(f"  Params shape: {params.shape}")
        # TODO: Add more validation checks

if __name__ == "__main__":
    print("[dataloader.py] Testing DataLoader...")
    
    # This will fail until you generate the dataset, but it's here for testing
    # try:
    #     loaders = create_dataloaders(batch_size=32)
    #     validate_dataloader(loaders['train'])
    # except Exception as e:
    #     print(f"Error: {e}")
    #     print("(This is expected if dataset hasn't been generated yet)")
