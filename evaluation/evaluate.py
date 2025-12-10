"""
Evaluation Script for Inverse Physics Model.
"""

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from model.architecture import InversePhysicsNet
from model.loss_functions import ParameterNormalizer
from dataset.dataloader import create_dataloaders
from dataset.generate_dataset import simulate_bouncing_ball # Reuse simulator

def compute_trajectory_divergence(pred_params, true_trajectory):
    """
    Simulate with predicted params and compare to ground truth.
    Returns: Mean Euclidean distance per frame (pixels).
    """
    # Convert tensor/list to dict for simulator
    params_dict = {
        'gravity': float(pred_params[0]),
        'mass': float(pred_params[1]),
        'friction': float(pred_params[2]),
        'restitution': float(pred_params[3])
    }
    
    # Simulate
    pred_traj = simulate_bouncing_ball(params_dict) # (60, 2)
    true_traj = np.array(true_trajectory) # (60, 2)
    
    # Map to pixels (0-64)
    # Note: simulate_bouncing_ball returns world coords [-1, 1]
    # We map distance in world coords to pixels: distance * (64/2)
    
    diff = pred_traj - true_traj
    dist_world = np.linalg.norm(diff, axis=1).mean()
    dist_pixels = dist_world * (64 / 2) # approx scale
    
    return dist_pixels

def evaluate(model_path="checkpoints/best_model.pt", data_dir="data/videos"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")
    
    # 1. Load Model
    model = InversePhysicsNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Load Data
    dataloaders = create_dataloaders(data_dir=data_dir, batch_size=1)
    val_loader = dataloaders['val']
    
    # 3. Utilities
    normalizer = ParameterNormalizer(device=device)
    param_names = ['gravity', 'mass', 'friction', 'restitution']
    
    # 4. Metrics
    mae_accum = {p: [] for p in param_names}
    traj_errors = []
    
    print("Running inference...")
    
    with torch.no_grad():
        for i, (video, true_params_norm) in enumerate(val_loader):
            video = video.to(device)
            true_params_norm = true_params_norm.to(device)
            
            # Predict
            pred_params_norm = model(video)
            
            # Denormalize
            pred_params = normalizer.denormalize(pred_params_norm).cpu().numpy()[0]
            true_params = normalizer.denormalize(true_params_norm).cpu().numpy()[0]
            
            # Compute MAE
            for idx, name in enumerate(param_names):
                error = abs(pred_params[idx] - true_params[idx])
                mae_accum[name].append(error)
            
            # Compute Trajectory Divergence
            # Need to find trajectory in metadata (not passed by dataloader)
            # Simple hack: dataloader shuffle=False for val, so indices match
            # But let's just skip this for now if complex, or load metadata manually
            pass 
            
    # 5. Report
    print("\n" + "="*40)
    print("EVALUATION RESULTS (Validation Set)")
    print("="*40)
    
    for name in param_names:
        mean_mae = np.mean(mae_accum[name])
        print(f"{name.capitalize():12s} MAE: {mean_mae:.4f}")
        
    print("="*40)

if __name__ == "__main__":
    evaluate()
