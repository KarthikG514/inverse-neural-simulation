"""
PyTorch DataLoader for Inverse Physics Dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

class InversePhysicsDataset(Dataset):
    """Dataset for physics parameter prediction from video."""
    
    def __init__(self, video_dir: str, metadata_path: str):
        """
        Args:
            video_dir: Path to videos folder
            metadata_path: Path to metadata.json
        """
        self.video_dir = Path(video_dir)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        print(f"[Dataset] Loaded {len(self.metadata)} samples from {metadata_path}")
        
        self.param_names = ['gravity', 'mass', 'friction', 'restitution']
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int):
        item = self.metadata[idx]
        
        # 1. Load Video
        # Handle path differences (Windows vs Linux)
        video_filename = Path(item['video_path']).name
        video_path = self.video_dir / video_filename
        
        video_tensor = self.load_video(str(video_path))
        
        # 2. Load Parameters
        params = item['params']
        param_tensor = torch.tensor([
            params[p] for p in self.param_names
        ], dtype=torch.float32)
        
        return video_tensor, param_tensor
    
    def load_video(self, video_path: str) -> torch.Tensor:
        """Load MP4, return (1, 60, 64, 64) tensor in [0, 1]."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            
        cap.release()
        
        # Ensure exactly 60 frames (pad or trim)
        if len(frames) < 60:
            # Pad with last frame
            last_frame = frames[-1] if frames else np.zeros((64,64), dtype=np.uint8)
            frames += [last_frame] * (60 - len(frames))
        elif len(frames) > 60:
            frames = frames[:60]
            
        # Stack -> (60, 64, 64)
        video = np.stack(frames, axis=0)
        
        # Normalize to [0, 1]
        video = video.astype(np.float32) / 255.0
        
        # Add channel dim -> (1, 60, 64, 64)
        video_tensor = torch.from_numpy(video).unsqueeze(0)
        
        return video_tensor

def create_dataloaders(data_dir="data/videos", batch_size=16, val_split=0.2):
    """Create train/val dataloaders."""
    dataset = InversePhysicsDataset(
        video_dir=data_dir,
        metadata_path=f"{data_dir}/metadata.json"
    )
    
    # Split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return {'train': train_loader, 'val': val_loader}

if __name__ == "__main__":
    # Test
    loaders = create_dataloaders(batch_size=4)
    train_loader = loaders['train']
    
    print(f"Train batches: {len(train_loader)}")
    
    # Get one batch
    videos, params = next(iter(train_loader))
    print(f"Batch videos shape: {videos.shape}")
    print(f"Batch params shape: {params.shape}")
    print(f"Video range: {videos.min():.2f} - {videos.max():.2f}")
    
    assert videos.shape == (4, 1, 60, 64, 64)
    assert params.shape == (4, 4)
    print("✓ DataLoader working!")
