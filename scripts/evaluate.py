"""
Evaluate trained model

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --config configs/ieee33_config.yaml

Author: Your Name
Date: 2026-01-18
"""

import sys
import os
# Add src directory to path (insert at beginning to avoid conflicts)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data.dataloader import get_dataloader
from models.graph_mamba import GraphMamba
from physics.constraints import PhysicsInformedLayer, PhysicsInformedGraphMamba
from utils.utils import load_config, load_checkpoint, MetricsCalculator


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str
) -> dict:
    """Comprehensive evaluation"""
    model.eval()

    all_pred_states = {'v_mag': [], 'v_ang': []}
    all_true_states = {'v_mag': [], 'v_ang': []}
    all_pred_params = {'r_line': [], 'x_line': []}
    all_true_params = {'r_line': [], 'x_line': []}
    all_physics_loss = []

    for batch in dataloader:
        # Move to device
        measurements = {k: v.to(device) for k, v in batch['measurements'].items()}
        obs_mask = batch['obs_mask'].to(device)
        true_states = {k: v.to(device) for k, v in batch['true_states'].items()}
        true_params = {k: v.to(device) for k, v in batch['parameters'].items()}
        edge_index = batch['topology']['edge_index'].to(device)
        edge_attr = batch['topology']['edge_attr'].to(device)

        # Forward pass
        pred_states, pred_params, physics_loss = model(
            measurements, edge_index, edge_attr, obs_mask
        )

        # Store results
        for key in ['v_mag', 'v_ang']:
            all_pred_states[key].append(pred_states[key].cpu())
            all_true_states[key].append(true_states[key].cpu())

        for key in ['r_line', 'x_line']:
            all_pred_params[key].append(pred_params[key].cpu())
            all_true_params[key].append(true_params[key].cpu())

        all_physics_loss.append(physics_loss.item())

    # Concatenate results
    pred_states = {k: torch.cat(v) for k, v in all_pred_states.items()}
    true_states = {k: torch.cat(v) for k, v in all_true_states.items()}
    pred_params = {k: torch.cat(v) for k, v in all_pred_params.items()}
    true_params = {k: torch.cat(v) for k, v in all_true_params.items()}

    # Compute metrics
    metrics = MetricsCalculator.compute_all_metrics(
        pred_states, true_states, pred_params, true_params
    )

    metrics['physics_loss_mean'] = np.mean(all_physics_loss)

    return metrics, pred_states, true_states, pred_params, true_params


def plot_results(pred_states, true_states, pred_params, true_params, save_dir):
    """Generate visualization plots"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Voltage magnitude comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    v_pred = pred_states['v_mag'].flatten().numpy()
    v_true = true_states['v_mag'].flatten().numpy()

    axes[0].scatter(v_true, v_pred, alpha=0.3, s=1)
    axes[0].plot([v_true.min(), v_true.max()], [v_true.min(), v_true.max()],
                 'r--', label='Perfect')
    axes[0].set_xlabel('True Voltage (p.u.)')
    axes[0].set_ylabel('Predicted Voltage (p.u.)')
    axes[0].set_title('Voltage Magnitude: Prediction vs Ground Truth')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    error = v_pred - v_true
    axes[1].hist(error, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error (p.u.)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Voltage Error Distribution\nMean: {error.mean():.6f}, Std: {error.std():.6f}')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'voltage_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Parameter estimation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    r_pred = pred_params['r_line'].flatten().numpy()
    r_true = true_params['r_line'].flatten().numpy()
    x_pred = pred_params['x_line'].flatten().numpy()
    x_true = true_params['x_line'].flatten().numpy()

    axes[0].scatter(r_true, r_pred, alpha=0.5, s=10, label='Resistance')
    axes[0].plot([r_true.min(), r_true.max()], [r_true.min(), r_true.max()],
                 'r--', label='Perfect')
    axes[0].set_xlabel('True R (Ω/km)')
    axes[0].set_ylabel('Predicted R (Ω/km)')
    axes[0].set_title('Line Resistance Estimation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(x_true, x_pred, alpha=0.5, s=10, label='Reactance', color='orange')
    axes[1].plot([x_true.min(), x_true.max()], [x_true.min(), x_true.max()],
                 'r--', label='Perfect')
    axes[1].set_xlabel('True X (Ω/km)')
    axes[1].set_ylabel('Predicted X (Ω/km)')
    axes[1].set_title('Line Reactance Estimation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'parameter_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Plots saved to {save_dir}")


def main(args):
    # Load config
    config = load_config(args.config)
    device = config['hardware']['device'] if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*60}")
    print(f"Evaluating {config['system']['name']} model")
    print(f"{'='*60}\n")

    # Load test data
    data_path = f"data/raw/{config['system']['name']}_dataset.pkl"
    test_loader = get_dataloader(
        data_path=data_path,
        batch_size=config['training']['batch_size'],
        split='test',
        num_workers=config['hardware']['num_workers'],
        sequence_length=10
    )

    # Build model
    graph_mamba = GraphMamba(
        num_nodes=config['system']['num_buses'],
        num_edges=len(test_loader.dataset.data[0]['topology']['edge_index'][0]),
        input_dim=3,
        spatial_config=config['model']['spatial_encoder'],
        temporal_config=config['model']['temporal_encoder'],
        state_head_config=config['model']['state_head'],
        parameter_head_config=config['model']['parameter_head']
    )

    physics_layer = PhysicsInformedLayer(
        constraint_type=config['physics']['constraint_type'],
        projection_method=config['physics']['projection_method'],
        max_iterations=config['physics']['max_iterations'],
        tolerance=config['physics']['tolerance'],
        voltage_limits=config['physics']['power_flow']['voltage_limits'],
        power_balance_weight=config['physics']['power_flow']['power_balance_weight']
    )

    model = PhysicsInformedGraphMamba(graph_mamba, physics_layer).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Evaluate
    print("\nEvaluating on test set...")
    metrics, pred_states, true_states, pred_params, true_params = evaluate_model(
        model, test_loader, device
    )

    # Print results
    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"{'='*60}")
    for key, val in metrics.items():
        print(f"  {key}: {val:.6f}")
    print(f"{'='*60}\n")

    # Generate plots
    if args.plot:
        plot_results(
            pred_states, true_states, pred_params, true_params,
            save_dir=Path(args.checkpoint).parent / 'evaluation_plots'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')

    args = parser.parse_args()
    main(args)
