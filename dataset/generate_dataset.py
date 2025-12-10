"""
Dataset Generator for Inverse Neural Simulation.

Generates synthetic videos of a bouncing ball using 2D physics equations.
Output: 64x64 grayscale videos + metadata JSON.
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

# Configuration
OUTPUT_DIR = Path("data/videos")
NUM_SAMPLES = 5000  # Start small for testing
IMG_SIZE = 64
FPS = 30
DURATION = 2.0  # seconds
NUM_FRAMES = int(DURATION * FPS)

def simulate_bouncing_ball(params: Dict[str, float]) -> np.ndarray:
    """
    Simulate a bouncing ball with given physics parameters.
    Returns: trajectory (NUM_FRAMES, 2) array of [x, y] positions.
    """
    gravity = params['gravity']       # m/s^2 (downward acceleration)
    friction = params['friction']     # Air resistance/friction (0-1)
    restitution = params['restitution'] # Bounciness (0-1)
    
    # Initial state
    pos = np.array([0.0, 0.8])  # Start near top center (x=0, y=0.8)
    vel = np.array([np.random.uniform(-0.5, 0.5), 0.0]) # Random horizontal push
    
    dt = 1.0 / FPS
    trajectory = []
    
    # Simulation loop (Euler integration)
    for _ in range(NUM_FRAMES):
        # 1. Update velocity
        vel[1] -= gravity * dt  # Gravity acts down
        vel *= (1.0 - friction * dt) # Friction slows it down
        
        # 2. Update position
        pos += vel * dt
        
        # 3. Collision detection (Ground is at y = -1.0)
        if pos[1] <= -0.9: # ball radius approx 0.1
            pos[1] = -0.9 # Hard constraint
            vel[1] *= -restitution # Bounce!
            
            # Stop bouncing if velocity is tiny (prevent infinite micro-bounces)
            if abs(vel[1]) < 0.5:
                vel[1] = 0
                
        # 4. Wall collisions (x = -1.0 and x = 1.0)
        if abs(pos[0]) > 0.9:
            pos[0] = np.sign(pos[0]) * 0.9
            vel[0] *= -1
            
        trajectory.append(pos.copy())
        
    return np.array(trajectory)

def render_video(trajectory: np.ndarray, output_path: str):
    """
    Render trajectory to MP4 video.
    """
    frames = []
    
    for pos in trajectory:
        # Create black canvas
        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        
        # Map world coords [-1, 1] to pixels [0, 64]
        # x: -1 -> 0, 1 -> 64
        # y: -1 -> 64, 1 -> 0 (image y is flipped)
        px = int((pos[0] + 1) / 2 * (IMG_SIZE - 1))
        py = int((1 - (pos[1] + 1) / 2) * (IMG_SIZE - 1))
        
        # Draw white ball
        cv2.circle(img, (px, py), radius=3, color=255, thickness=-1)
        frames.append(img)
    
    # Save as MP4
    height, width = IMG_SIZE, IMG_SIZE
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1' on some systems
    out = cv2.VideoWriter(str(output_path), fourcc, FPS, (width, height), isColor=False)
    
    for frame in frames:
        out.write(frame)
    out.release()

def generate_dataset(num_samples: int = NUM_SAMPLES):
    """Generate videos and metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []
    
    print(f"Generating {num_samples} samples...")
    
    for i in tqdm(range(num_samples)):
        # 1. Randomize physics
        params = {
            'gravity': np.random.uniform(5.0, 15.0),
            'mass': np.random.uniform(0.5, 5.0), # Mass doesn't affect kinematics in simple vacuum gravity, but we track it
            'friction': np.random.uniform(0.0, 0.5),
            'restitution': np.random.uniform(0.3, 0.95)
        }
        
        # 2. Simulate
        trajectory = simulate_bouncing_ball(params)
        
        # 3. Render
        filename = f"video_{i:05d}.mp4"
        filepath = OUTPUT_DIR / filename
        render_video(trajectory, filepath)
        
        # 4. Save metadata
        metadata.append({
            'video_path': str(filepath),
            'params': params,
            'trajectory': trajectory.tolist() # Save ground truth positions
        })
    
    # Save JSON
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Dataset generated at {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_dataset()
