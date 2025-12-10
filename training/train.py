"""
Training loop for Inverse Neural Simulation.

Trains InversePhysicsNet to predict normalized physics parameters
from video clips of a bouncing ball.
"""

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.architecture import InversePhysicsNet
from model.loss_functions import ParameterRegressionLoss
from dataset.dataloader import create_dataloaders


def train(args):
    # ----- Device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # ----- Data -----
    print("[train] Loading data...")
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # ----- Model -----
    print("[train] Creating model...")
    model = InversePhysicsNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Total parameters: {total_params:,}")

    # ----- Loss & Optimizer -----
    loss_fn = ParameterRegressionLoss(device=str(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ----- Tracking -----
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] Starting training for {args.epochs} epochs...")

    # ----- Epoch Loop -----
    for epoch in range(args.epochs):
        # === TRAIN ===
        model.train()
        running_train_loss = 0.0

        for videos, params_norm in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs} [train]",
            leave=False,
        ):
            videos = videos.to(device)          # (B, 1, 60, 64, 64)
            params_norm = params_norm.to(device)  # (B, 4) in [0,1]

            # Forward
            pred = model(videos)                # (B, 4)
            loss = loss_fn(pred, params_norm)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / max(len(train_loader), 1)
        train_losses.append(avg_train_loss)

        # === VALIDATE ===
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for videos, params_norm in tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{args.epochs} [val]",
                leave=False,
            ):
                videos = videos.to(device)
                params_norm = params_norm.to(device)

                pred = model(videos)
                loss = loss_fn(pred, params_norm)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / max(len(val_loader), 1)
        val_losses.append(avg_val_loss)

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  ★ New best model saved to {best_path}")

    # ----- Plot Loss Curves -----
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (normalized params)")
    plt.title("Training / Validation Loss")
    plt.legend()
    loss_plot_path = output_dir / "loss_curve.png"
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"[train] Saved loss curve to {loss_plot_path}")
    print("[train] Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Inverse Physics Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--data_dir", type=str, default="data/videos", help="Directory with videos"
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2, help="Fraction of data for validation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
