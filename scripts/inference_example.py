"""
Example: Using the trained model for inference

This script demonstrates how to use a trained Graph Mamba model
for real-time state estimation on new data.
"""

import sys
sys.path.append('src')

import torch
import numpy as np
from pathlib import Path

from models.graph_mamba import GraphMamba
from physics.constraints import PhysicsInformedLayer, PhysicsInformedGraphMamba
from utils.utils import load_config


def load_trained_model(checkpoint_path: str, config_path: str, device: str = 'cuda'):
    """Load a trained model from checkpoint"""

    # Load config
    config = load_config(config_path)

    # Build model architecture
    graph_mamba = GraphMamba(
        num_nodes=config['system']['num_buses'],
        num_edges=32,  # Adjust based on your system
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

    model = PhysicsInformedGraphMamba(graph_mamba, physics_layer)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    print(f"✓ Training loss: {checkpoint['loss']:.6f}")

    return model, config


@torch.no_grad()
def predict_single_timestep(
    model,
    voltage_measurements: np.ndarray,  # [num_buses]
    power_measurements: np.ndarray,     # [num_buses, 2] (P, Q)
    edge_index: np.ndarray,             # [2, num_edges]
    edge_attr: np.ndarray,              # [num_edges, 2] (R, X)
    observation_mask: np.ndarray = None, # [num_buses]
    device: str = 'cuda'
):
    """
    Make prediction for a single time step

    Args:
        voltage_measurements: Measured voltage magnitudes [num_buses]
        power_measurements: Measured active/reactive power [num_buses, 2]
        edge_index: Graph edges [2, num_edges]
        edge_attr: Edge attributes (impedances) [num_edges, 2]
        observation_mask: Which buses are observed [num_buses]

    Returns:
        estimated_states: Dict with 'v_mag', 'v_ang'
        estimated_parameters: Dict with 'r_line', 'x_line'
    """

    num_buses = voltage_measurements.shape[0]

    # Create dummy sequence (use last measurement repeated)
    sequence_length = 10
    measurements = {
        'v_mag': torch.from_numpy(
            np.repeat(voltage_measurements[None, None, :], sequence_length, axis=1)
        ).float().to(device),
        'p_bus': torch.from_numpy(
            np.repeat(power_measurements[:, 0][None, None, :], sequence_length, axis=1)
        ).float().to(device),
        'q_bus': torch.from_numpy(
            np.repeat(power_measurements[:, 1][None, None, :], sequence_length, axis=1)
        ).float().to(device),
    }

    # Observation mask
    if observation_mask is None:
        observation_mask = np.ones(num_buses, dtype=bool)

    obs_mask = torch.from_numpy(
        np.repeat(observation_mask[None, None, :], sequence_length, axis=1)
    ).bool().to(device)

    # Edge info
    edge_index_t = torch.from_numpy(edge_index).long().to(device)
    edge_attr_t = torch.from_numpy(edge_attr).float().to(device)

    # Predict
    states, parameters, physics_loss = model(
        measurements, edge_index_t, edge_attr_t, obs_mask
    )

    # Convert to numpy
    estimated_states = {
        'v_mag': states['v_mag'][0].cpu().numpy(),
        'v_ang': states['v_ang'][0].cpu().numpy()
    }

    estimated_parameters = {
        'r_line': parameters['r_line'][0].cpu().numpy(),
        'x_line': parameters['x_line'][0].cpu().numpy()
    }

    return estimated_states, estimated_parameters, physics_loss.item()


def main():
    """Example usage"""

    # Paths
    checkpoint_path = "checkpoints/ieee33/best_model.pt"
    config_path = "configs/ieee33_config.yaml"

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, config = load_trained_model(checkpoint_path, config_path, device)

    print("\n" + "="*60)
    print("Running inference example...")
    print("="*60 + "\n")

    # Create synthetic measurement data (in practice, this comes from PMUs/SCADA)
    num_buses = 33
    num_edges = 32

    # Voltage measurements (with some missing)
    voltage_measurements = np.random.rand(num_buses) * 0.1 + 0.95  # ~1.0 p.u.

    # Power measurements
    p_measurements = np.random.randn(num_buses) * 0.5  # MW
    q_measurements = np.random.randn(num_buses) * 0.3  # MVAr
    power_measurements = np.stack([p_measurements, q_measurements], axis=1)

    # Observation mask (only 30% buses have PMUs)
    observation_mask = np.random.rand(num_buses) < 0.3
    observation_mask[0] = True  # Slack bus always observed

    # Graph topology (simple radial for demo)
    edge_index = np.array([
        list(range(num_edges)),
        list(range(1, num_edges + 1))
    ])

    # Line impedances
    edge_attr = np.random.rand(num_edges, 2) * 0.5  # Ω/km

    print(f"Input:")
    print(f"  Buses: {num_buses}")
    print(f"  Observed buses: {observation_mask.sum()}/{num_buses}")
    print(f"  Voltage range: [{voltage_measurements.min():.3f}, {voltage_measurements.max():.3f}] p.u.")
    print()

    # Predict
    states, parameters, phys_loss = predict_single_timestep(
        model,
        voltage_measurements,
        power_measurements,
        edge_index,
        edge_attr,
        observation_mask,
        device
    )

    print("Output:")
    print(f"  Estimated voltage magnitude: [{states['v_mag'].min():.3f}, {states['v_mag'].max():.3f}] p.u.")
    print(f"  Estimated voltage angle: [{states['v_ang'].min():.3f}, {states['v_ang'].max():.3f}] rad")
    print(f"  Estimated line resistance: {parameters['r_line'].mean():.4f} ± {parameters['r_line'].std():.4f} Ω/km")
    print(f"  Estimated line reactance: {parameters['x_line'].mean():.4f} ± {parameters['x_line'].std():.4f} Ω/km")
    print(f"  Physics constraint loss: {phys_loss:.6f}")
    print()

    # Check physical feasibility
    voltage_ok = (states['v_mag'] >= 0.95).all() and (states['v_mag'] <= 1.05).all()
    params_ok = (parameters['r_line'] > 0).all() and (parameters['x_line'] > 0).all()

    print("Physical Feasibility Check:")
    print(f"  ✓ Voltages within limits: {voltage_ok}")
    print(f"  ✓ Parameters positive: {params_ok}")

    if voltage_ok and params_ok and phys_loss < 0.01:
        print("\n✓ All checks passed! Estimation successful.")
    else:
        print("\n⚠ Warning: Some constraints violated. Check input data quality.")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
