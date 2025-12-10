import torch
import numpy as np
from pathlib import Path

from model.architecture import InversePhysicsNet
from model.loss_functions import ParameterNormalizer
from dataset.dataloader import create_dataloaders

def debug_predictions(model_path="checkpoints/best_model.pt", num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[debug] Using device: {device}")

    # Load model
    model = InversePhysicsNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load data (batch_size=1 for easier inspection)
    loaders = create_dataloaders(batch_size=1)
    val_loader = loaders["val"]

    normalizer = ParameterNormalizer(device=device)
    param_names = ["gravity", "mass", "friction", "restitution"]

    print("[debug] Showing a few predictions vs ground truth (original scale):")
    print("-" * 60)

    with torch.no_grad():
        for i, (video, true_norm) in enumerate(val_loader):
            if i >= num_samples:
                break

            video = video.to(device)
            true_norm = true_norm.to(device)

            pred_norm = model(video)                          # (1,4), normalized
            pred = normalizer.denormalize(pred_norm)[0].cpu().numpy()
            true = normalizer.denormalize(true_norm)[0].cpu().numpy()

            print(f"Sample {i}:")
            for j, name in enumerate(param_names):
                print(f"  {name:11s} | true: {true[j]:8.3f} | pred: {pred[j]:8.3f} | diff: {abs(pred[j]-true[j]):8.3f}")
            print("-" * 60)

if __name__ == "__main__":
    debug_predictions()
