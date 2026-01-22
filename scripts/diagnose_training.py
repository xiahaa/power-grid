"""
Diagnostic script to identify training issues

Author: Assistant
Date: 2026-01-21
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from data.dataloader import get_dataloader
from models.graph_mamba import GraphMamba
from physics.constraints import PhysicsInformedLayer, PhysicsInformedGraphMamba
from train.loss import JointEstimationLoss
from utils.utils import load_config

def diagnose_gradients(model, batch, criterion, device):
    """Check for gradient flow issues"""
    model.train()

    # Move batch to device
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

    # Compute loss
    loss, loss_dict = criterion(
        pred_states, true_states,
        pred_params, true_params,
        physics_loss,
        obs_mask
    )

    # Backward pass
    loss.backward()

    # Check gradients
    print("\n=== Gradient Analysis ===")
    zero_grad_params = []
    large_grad_params = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            if param_norm > 1e-8:
                ratio = grad_norm / param_norm
                print(f"{name:50s} | grad_norm: {grad_norm:.6f} | param_norm: {param_norm:.6f} | ratio: {ratio:.6f}")
                if ratio > 100:
                    large_grad_params.append((name, ratio))
            else:
                print(f"{name:50s} | grad_norm: {grad_norm:.6f} | param_norm: {param_norm:.6f} | ZERO PARAM!")
                zero_grad_params.append(name)

            if grad_norm < 1e-8:
                zero_grad_params.append(name)
        else:
            print(f"{name:50s} | NO GRADIENT")

    if zero_grad_params:
        print(f"\n⚠️  WARNING: {len(zero_grad_params)} parameters with zero or near-zero gradients")
    if large_grad_params:
        print(f"\n⚠️  WARNING: {len(large_grad_params)} parameters with very large gradient ratios (>100)")
        for name, ratio in large_grad_params[:5]:
            print(f"    {name}: {ratio:.2f}")

    # Check output statistics
    print("\n=== Output Statistics ===")
    print(f"v_mag - mean: {pred_states['v_mag'].mean():.4f}, std: {pred_states['v_mag'].std():.4f}, min: {pred_states['v_mag'].min():.4f}, max: {pred_states['v_mag'].max():.4f}")
    print(f"v_ang - mean: {pred_states['v_ang'].mean():.4f}, std: {pred_states['v_ang'].std():.4f}, min: {pred_states['v_ang'].min():.4f}, max: {pred_states['v_ang'].max():.4f}")
    print(f"r_line - mean: {pred_params['r_line'].mean():.4f}, std: {pred_params['r_line'].std():.4f}, min: {pred_params['r_line'].min():.4f}, max: {pred_params['r_line'].max():.4f}")
    print(f"x_line - mean: {pred_params['x_line'].mean():.4f}, std: {pred_params['x_line'].std():.4f}, min: {pred_params['x_line'].min():.4f}, max: {pred_params['x_line'].max():.4f}")

    print("\n=== Target Statistics ===")
    print(f"v_mag - mean: {true_states['v_mag'].mean():.4f}, std: {true_states['v_mag'].std():.4f}, min: {true_states['v_mag'].min():.4f}, max: {true_states['v_mag'].max():.4f}")
    print(f"v_ang - mean: {true_states['v_ang'].mean():.4f}, std: {true_states['v_ang'].std():.4f}, min: {true_states['v_ang'].min():.4f}, max: {true_states['v_ang'].max():.4f}")
    print(f"r_line - mean: {true_params['r_line'].mean():.4f}, std: {true_params['r_line'].std():.4f}, min: {true_params['r_line'].min():.4f}, max: {true_params['r_line'].max():.4f}")
    print(f"x_line - mean: {true_params['x_line'].mean():.4f}, std: {true_params['x_line'].std():.4f}, min: {true_params['x_line'].min():.4f}, max: {true_params['x_line'].max():.4f}")

    print("\n=== Loss Components ===")
    for key, val in loss_dict.items():
        print(f"{key}: {val:.6f}")

    return loss_dict

if __name__ == "__main__":
    # Load config
    config = load_config("configs/ieee33_config.yaml")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    train_loader = get_dataloader(
        data_path="data/raw/ieee33_dataset.pkl",
        batch_size=4,
        split='train',
        num_workers=0,
        sequence_length=10
    )

    # Build model
    graph_mamba = GraphMamba(
        num_nodes=config['system']['num_buses'],
        num_edges=len(train_loader.dataset.data[0]['topology']['edge_index'][0]),
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

    # Loss function
    criterion = JointEstimationLoss(
        state_weight=config['loss']['state_weight'],
        parameter_weight=config['loss']['parameter_weight'],
        physics_weight=config['loss']['physics_weight'],
        temporal_smoothness=config['loss']['temporal_smoothness']
    )

    # Get one batch
    batch = next(iter(train_loader))

    # Run diagnosis
    diagnose_gradients(model, batch, criterion, device)
