"""
Generate power grid dataset

Usage:
    python scripts/generate_data.py --system ieee33 --num_scenarios 1000

Author: Your Name
Date: 2026-01-18
"""

import sys
import os
# Add src directory to path (insert at beginning to avoid conflicts)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
from pathlib import Path
from data.data_generator import PowerGridDataGenerator


def main(args):
    print(f"\n{'='*60}")
    print(f"Generating {args.system} Dataset")
    print(f"{'='*60}\n")

    # Create generator
    generator = PowerGridDataGenerator(
        system_name=args.system,
        num_scenarios=args.num_scenarios,
        time_steps=args.time_steps,
        pmu_coverage=args.pmu_coverage,
        noise_std=args.noise_std,
        parameter_drift_enabled=args.parameter_drift,
        pv_penetration=args.pv_penetration,
        seed=args.seed
    )

    # Generate dataset
    save_path = Path(args.output) / f"{args.system}_dataset.pkl"
    dataset = generator.generate_dataset(save_path=str(save_path))

    print("\n✓ Dataset generation complete!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate power grid dataset for training'
    )
    parser.add_argument('--system', type=str, default='ieee33',
                       choices=['ieee33', 'ieee118'],
                       help='Power system to simulate')
    parser.add_argument('--num_scenarios', type=int, default=1000,
                       help='Number of scenarios')
    parser.add_argument('--time_steps', type=int, default=288,
                       help='Time steps per scenario (288 = 24h @ 5min)')
    parser.add_argument('--pmu_coverage', type=float, default=0.3,
                       help='PMU coverage ratio')
    parser.add_argument('--noise_std', type=float, default=0.02,
                       help='Measurement noise std')
    parser.add_argument('--parameter_drift', action='store_true',
                       help='Enable parameter drift')
    parser.add_argument('--pv_penetration', type=float, default=0.4,
                       help='PV penetration ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default='data/raw',
                       help='Output directory')

    args = parser.parse_args()
    main(args)
