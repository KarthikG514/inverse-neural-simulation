"""
Create Demo Video: Input vs Predicted Simulation.
"""

import torch
import numpy as np
import cv2
import json
from pathlib import Path
from model.architecture import InversePhysicsNet
from model.loss_functions import ParameterNormalizer
from dataset.dataloader import create_dataloaders
from dataset.generate_dataset import simulate_bouncing_ball

def render_frame(pos, img_size=64):
    """Render a single dot at pos (world coords)."""
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    px = int((pos[0] + 1) / 2 * (img_size - 1))
    py = int((1 - (pos[1] + 1) / 2) * (img_size - 1))
    cv2.circle(img, (px, py), radius=3, color=255, thickness=-1)
    return img

def create_demo(model_path="checkpoints/best_model.pt", output_path="demo/comparison.mp4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating demo on {device}...")
    
    # 1. Load Model
    model = InversePhysicsNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Get 1 Sample
    dataloaders = create_dataloaders(batch_size=1)
    video_tensor, true_params_norm = next(iter(dataloaders['val']))
    
    # 3. Predict
    with torch.no_grad():
        pred_params_norm = model(video_tensor.to(device))
    
    # 4. Denormalize
    normalizer = ParameterNormalizer(device=device)
    pred_params = normalizer.denormalize(pred_params_norm).cpu().numpy()[0]
    true_params = normalizer.denormalize(true_params_norm.to(device)).cpu().numpy()[0]
    
    print(f"True Params: {true_params}")
    print(f"Pred Params: {pred_params}")
    
    # 5. Simulate with Predicted Params
    params_dict = {
        'gravity': float(pred_params[0]),
        'mass': float(pred_params[1]),
        'friction': float(pred_params[2]),
        'restitution': float(pred_params[3])
    }
    pred_traj = simulate_bouncing_ball(params_dict)
    
    # 6. Stitch Video
    input_video = video_tensor[0, 0].numpy() # (60, 64, 64)
    frames = []
    
    for i in range(len(input_video)):
        # Input frame (convert 0-1 to 0-255)
        frame_in = (input_video[i] * 255).astype(np.uint8)
        
        # Predicted frame
        frame_pred = render_frame(pred_traj[i])
        
        # Combine Side-by-Side
        combined = np.hstack((frame_in, frame_pred))
        
        # Add Text
        cv2.putText(combined, "Input", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
        cv2.putText(combined, "Predicted", (74, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
        
        frames.append(combined)
    
    # 7. Save
    height, width = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), isColor=False)
    for f in frames:
        out.write(f)
    out.release()
    print(f"Saved demo to {output_path}")

if __name__ == "__main__":
    Path("demo").mkdir(exist_ok=True)
    create_demo()
