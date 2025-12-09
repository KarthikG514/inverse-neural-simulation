"""
Evaluation Script for Physics Parameter Prediction Model.

Metrics computed:
1. Parameter MAE: How close are predicted parameters to ground truth?
2. Trajectory Divergence: How close is simulated trajectory to input video?
3. Per-parameter accuracy breakdown
"""

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from model.architecture import InversePhysicsNet
from model.loss_functions import ParameterNormalizer, PARAM_RANGES
from dataset.dataloader import create_dataloaders

class ModelEvaluator:
    """Evaluate physics prediction model on test set."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize evaluator with trained model.
        
        Args:
            model_path: Path to best_model.pt
            device: 'cuda' or 'cpu'
        
        TODO:
        - Load model
        - Load checkpoint state_dict
        - Set to eval mode
        - Initialize normalizer
        """
        self.device = torch.device(device)
        self.model = InversePhysicsNet().to(self.device)
        self.normalizer = ParameterNormalizer()
        
        # TODO: Load checkpoint
        # checkpoint = torch.load(model_path, map_location=self.device)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        self.param_names = ['gravity', 'mass', 'friction', 'restitution']
        print(f"[ModelEvaluator] Model loaded from {model_path}")
    
    def compute_parameter_mae(self, predictions: np.ndarray,
                              ground_truth: np.ndarray) -> dict:
        """
        Compute Mean Absolute Error for each parameter.
        
        Args:
            predictions: (N, 4) predicted parameters (original scale)
            ground_truth: (N, 4) ground truth (original scale)
        
        Returns:
            dict with MAE for each parameter
        
        TODO:
        - For each of 4 parameters:
            mae = mean(|pred - true|)
        - Return dict with param names as keys
        """
        mae_dict = {}
        
        # TODO: Implement
        print("[compute_parameter_mae] Computing MAE for each parameter...")
        
        return mae_dict
    
    def evaluate_on_testset(self, test_loader):
        """
        Full evaluation on test set.
        
        Args:
            test_loader: DataLoader for test set
        
        Returns:
            dict with all metrics
        
        TODO:
        - Iterate through test_loader
        - Run inference (no gradients)
        - Denormalize predictions
        - Compute parameter MAE
        - (Optional: compute trajectory divergence with simulator)
        - Collect all metrics
        - Return summary stats
        """
        print("[evaluate_on_testset] Starting evaluation...")
        
        all_predictions = []
        all_ground_truth = []
        
        with torch.no_grad():
            # TODO: Iterate through test_loader
            # for videos, true_params in test_loader:
            #     videos = videos.to(self.device)
            #     true_params = true_params.to(self.device)
            #     
            #     # Forward pass
            #     pred_params_norm = self.model(videos)
            #     
            #     # Denormalize
            #     pred_params = self.normalizer.denormalize(pred_params_norm)
            #     true_params_denorm = self.normalizer.denormalize(true_params)
            #     
            #     # Collect
            #     all_predictions.append(pred_params.cpu().numpy())
            #     all_ground_truth.append(true_params_denorm.cpu().numpy())
            pass
        
        # Combine all batches
        # all_predictions = np.concatenate(all_predictions, axis=0)
        # all_ground_truth = np.concatenate(all_ground_truth, axis=0)
        
        # Compute metrics
        mae = self.compute_parameter_mae(all_predictions, all_ground_truth)
        
        results = {
            'parameter_mae': mae,
            'num_samples': len(all_ground_truth),
        }
        
        return results
    
    def create_evaluation_report(self, results: dict):
        """
        Create printable evaluation report.
        
        Args:
            results: Dict with all metrics
        
        TODO:
        - Format results as nice table
        - Print summary statistics
        - Save to JSON file
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        if 'parameter_mae' in results:
            print("\nParameter Prediction Accuracy (MAE):")
            for param_name, mae_value in results['parameter_mae'].items():
                print(f"  {param_name:12s}: {mae_value:.4f}")
        
        print(f"\nNum samples evaluated: {results.get('num_samples', 'N/A')}")
        print("="*60)

def main(model_path: str = "checkpoints/best_model.pt", 
         output_dir: str = "evaluation"):
    """
    Evaluate model and create report.
    
    Args:
        model_path: Path to trained model
        output_dir: Where to save results
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] Device: {device}")
    
    # Load test set
    print("[main] Loading test set...")
    # TODO: dataloaders = create_dataloaders(batch_size=32, ...)
    # test_loader = dataloaders['test']
    
    # Evaluate
    print("[main] Evaluating model...")
    evaluator = ModelEvaluator(model_path, device=device)
    # TODO: results = evaluator.evaluate_on_testset(test_loader)
    
    # Create report
    print("[main] Creating report...")
    # TODO: evaluator.create_evaluation_report(results)
    
    # Save results
    # TODO: output_dir = Path(output_dir)
    # output_dir.mkdir(exist_ok=True, parents=True)
    # with open(output_dir / "results.json", "w") as f:
    #     json.dump(results, f, indent=2)
    
    print(f"[main] Results saved to {output_dir}")

if __name__ == "__main__":
    main()
