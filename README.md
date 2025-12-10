# inverse-neural-simulation
Predicting physics parameters from video using deep learning
# 1) Generate dataset
python -m dataset.generate_dataset

# 2) Train model
python -m training.train --epochs 20 --batch_size 32 --lr 1e-3

# 3) Evaluate model
python -m evaluation.evaluate

# 4) Create demo video
python -m demo.create_demo

Dataset: 1000 synthetic videos (bouncing ball)
Model: InversePhysicsNet (3D CNN + MLP)
Validation MAE (approx.):
Gravity: ~8.4
Mass: ~5.4
Friction: ~0.12
Restitution: ~0.14

Synthetic dataset generator creates bouncing‑ball videos with random physics parameters.
A 3D CNN (InversePhysicsNet) encodes the video into features.
An MLP head regresses normalized physics parameters, which are then denormalized.
Evaluation compares predicted vs true parameters on a validation split.
Demo re‑simulates motion using predicted parameters and renders a comparison video.