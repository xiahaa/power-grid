"""
Robustness testing for Graph Mamba model

Tests model performance under:
1. Missing measurements (sparse PMU coverage)
2. Topology changes (line outages)
3. Bad data injection
4. Cascading failures (large systems)

Usage:
    python scripts/test_robustness.py --checkpoint checkpoints/best_model.pt --config configs/ieee33_config.yaml

Author: Your Name
Date: 2026-01-18
"""

import sys
sys.path.append('src')

import torch
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.dataloader import get_dataloader
from models.graph_mamba import GraphMamba
from physics.constraints import PhysicsInformedLayer, PhysicsInformedGraphMamba
from utils.utils import load_config, MetricsCalculator


class RobustnessTests:
    """Comprehensive robustness testing suite"""

    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def test_missing_measurements(self, missing_ratios=[0.0, 0.2, 0.4, 0.6, 0.8]):
        """
        Test with varying levels of missing measurements

        Simulates sensor failures or sparse PMU deployment
        """
        print("\n" + "="*60)
        print("Test 1: Missing Measurements")
        print("="*60)

        results = {}

        for ratio in missing_ratios:
            print(f"\nMissing ratio: {ratio*100:.0f}%")

            all_errors = []

            for batch in tqdm(self.dataloader, desc=f"Missing {ratio*100:.0f}%"):
                measurements = {k: v.to(self.device) for k, v in batch['measurements'].items()}
                true_states = {k: v.to(self.device) for k, v in batch['true_states'].items()}
                edge_index = batch['topology']['edge_index'].to(self.device)
                edge_attr = batch['topology']['edge_attr'].to(self.device)

                # Create degraded observation mask
                obs_mask = batch['obs_mask'].to(self.device)
                batch_size, seq_len, num_nodes = obs_mask.shape

                # Randomly drop additional measurements
                drop_mask = torch.rand_like(obs_mask.float()) > ratio
                degraded_mask = obs_mask & drop_mask

                # Zero out missing measurements
                for key in measurements.keys():
                    measurements[key] = measurements[key] * degraded_mask.float()

                # Predict
                pred_states, pred_params, _ = self.model(
                    measurements, edge_index, edge_attr, degraded_mask
                )

                # Compute error
                error = torch.abs(pred_states['v_mag'] - true_states['v_mag']).mean().item()
                all_errors.append(error)

            mean_error = np.mean(all_errors)
            results[ratio] = mean_error
            print(f"  Mean V_mag error: {mean_error:.6f} p.u.")

        return results

    @torch.no_grad()
    def test_topology_change(self, num_outages_list=[0, 1, 2, 3, 5]):
        """
        Test with line outages (topology changes)

        Simulates equipment failures or maintenance
        """
        print("\n" + "="*60)
        print("Test 2: Topology Changes (Line Outages)")
        print("="*60)

        results = {}

        for num_outages in num_outages_list:
            print(f"\nNumber of line outages: {num_outages}")

            all_errors = []

            for batch in tqdm(self.dataloader, desc=f"{num_outages} outages"):
                measurements = {k: v.to(self.device) for k, v in batch['measurements'].items()}
                true_states = {k: v.to(self.device) for k, v in batch['true_states'].items()}
                edge_index = batch['topology']['edge_index'].to(self.device).clone()
                edge_attr = batch['topology']['edge_attr'].to(self.device).clone()
                obs_mask = batch['obs_mask'].to(self.device)

                if num_outages > 0:
                    # Randomly remove lines (set impedance to very high)
                    num_edges = edge_index.shape[1]
                    outage_indices = np.random.choice(
                        num_edges,
                        size=min(num_outages, num_edges // 2),
                        replace=False
                    )
                    # Increase impedance to simulate disconnection
                    edge_attr[outage_indices] *= 1000.0

                # Predict
                pred_states, pred_params, _ = self.model(
                    measurements, edge_index, edge_attr, obs_mask
                )

                # Compute error
                error = torch.abs(pred_states['v_mag'] - true_states['v_mag']).mean().item()
                all_errors.append(error)

            mean_error = np.mean(all_errors)
            results[num_outages] = mean_error
            print(f"  Mean V_mag error: {mean_error:.6f} p.u.")

        return results

    @torch.no_grad()
    def test_bad_data(self, corruption_ratios=[0.0, 0.05, 0.1, 0.2, 0.3]):
        """
        Test with bad data injection (corrupted measurements)

        Simulates cyber-attacks or sensor malfunctions
        """
        print("\n" + "="*60)
        print("Test 3: Bad Data Injection")
        print("="*60)

        results = {}

        for ratio in corruption_ratios:
            print(f"\nCorruption ratio: {ratio*100:.0f}%")

            all_errors = []

            for batch in tqdm(self.dataloader, desc=f"Corrupt {ratio*100:.0f}%"):
                measurements = {k: v.to(self.device) for k, v in batch['measurements'].items()}
                true_states = {k: v.to(self.device) for k, v in batch['true_states'].items()}
                edge_index = batch['topology']['edge_index'].to(self.device)
                edge_attr = batch['topology']['edge_attr'].to(self.device)
                obs_mask = batch['obs_mask'].to(self.device)

                # Inject bad data
                if ratio > 0:
                    for key in measurements.keys():
                        batch_size, seq_len, num_nodes = measurements[key].shape

                        # Randomly corrupt measurements
                        corrupt_mask = torch.rand_like(measurements[key]) < ratio
                        # Add large random noise
                        noise = torch.randn_like(measurements[key]) * 0.5
                        measurements[key] = torch.where(
                            corrupt_mask,
                            measurements[key] + noise,
                            measurements[key]
                        )

                # Predict
                pred_states, pred_params, _ = self.model(
                    measurements, edge_index, edge_attr, obs_mask
                )

                # Compute error
                error = torch.abs(pred_states['v_mag'] - true_states['v_mag']).mean().item()
                all_errors.append(error)

            mean_error = np.mean(all_errors)
            results[ratio] = mean_error
            print(f"  Mean V_mag error: {mean_error:.6f} p.u.")

        return results

    def plot_robustness_results(self, results_dict, save_dir):
        """Generate robustness test visualizations"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Missing measurements
        if 'missing_measurements' in results_dict:
            data = results_dict['missing_measurements']
            x = [k*100 for k in data.keys()]
            y = list(data.values())
            axes[0].plot(x, y, 'o-', linewidth=2, markersize=8)
            axes[0].set_xlabel('Missing Measurements (%)', fontsize=12)
            axes[0].set_ylabel('Mean V_mag Error (p.u.)', fontsize=12)
            axes[0].set_title('Robustness to Missing Data', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)

        # 2. Topology changes
        if 'topology_change' in results_dict:
            data = results_dict['topology_change']
            x = list(data.keys())
            y = list(data.values())
            axes[1].plot(x, y, 's-', linewidth=2, markersize=8, color='orange')
            axes[1].set_xlabel('Number of Line Outages', fontsize=12)
            axes[1].set_ylabel('Mean V_mag Error (p.u.)', fontsize=12)
            axes[1].set_title('Robustness to Topology Changes', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)

        # 3. Bad data
        if 'bad_data' in results_dict:
            data = results_dict['bad_data']
            x = [k*100 for k in data.keys()]
            y = list(data.values())
            axes[2].plot(x, y, '^-', linewidth=2, markersize=8, color='red')
            axes[2].set_xlabel('Corrupted Measurements (%)', fontsize=12)
            axes[2].set_ylabel('Mean V_mag Error (p.u.)', fontsize=12)
            axes[2].set_title('Robustness to Bad Data', fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'robustness_tests.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Robustness plots saved to {save_dir}")


def main(args):
    # Load config
    config = load_config(args.config)
    device = config['hardware']['device'] if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*60}")
    print(f"Robustness Testing: {config['system']['name']}")
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
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}\n")

    # Run robustness tests
    tester = RobustnessTests(model, test_loader, device)

    results = {}

    if args.test_missing or args.all:
        results['missing_measurements'] = tester.test_missing_measurements()

    if args.test_topology or args.all:
        results['topology_change'] = tester.test_topology_change()

    if args.test_bad_data or args.all:
        results['bad_data'] = tester.test_bad_data()

    # Plot results
    if args.plot and results:
        tester.plot_robustness_results(
            results,
            save_dir=Path(args.checkpoint).parent / 'robustness_plots'
        )

    print("\n" + "="*60)
    print("✓ Robustness testing complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--test_missing', action='store_true',
                       help='Test missing measurements')
    parser.add_argument('--test_topology', action='store_true',
                       help='Test topology changes')
    parser.add_argument('--test_bad_data', action='store_true',
                       help='Test bad data injection')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')

    args = parser.parse_args()

    # If no specific test selected, run all
    if not (args.test_missing or args.test_topology or args.test_bad_data):
        args.all = True

    main(args)
