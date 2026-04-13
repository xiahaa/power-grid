"""
Data generation script for Multi-Rate Mamba Fusion DSSE (Phase 2)

Usage:
    python scripts/generate_data_v2.py --config configs/ieee123_config.yaml
    python scripts/generate_data_v2.py --config configs/ieee123_config.yaml --output data/my_dataset.pkl --num_scenarios 500
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
from pathlib import Path

from utils.utils import load_config
from data.data_generator_v2 import PowerGridDataGeneratorV2


def main(args):
    config = load_config(args.config)
    sys_cfg = config["system"]
    data_cfg = config["data"]
    meas_cfg = config["measurements"]
    topo_cfg = data_cfg.get("topology_change", {})

    num_scenarios = args.num_scenarios or data_cfg["num_scenarios"]
    output_path = args.output or "data/%s_dataset.pkl" % sys_cfg["name"]

    print("Generating dataset for %s" % sys_cfg["name"])
    print("  Scenarios: %d" % num_scenarios)
    print("  Time steps: %d" % data_cfg["time_steps"])
    print("  PMU coverage: %.1f%%" % (sys_cfg["pmu_coverage"] * 100))
    print("  SCADA coverage: %.1f%%" % (sys_cfg["scada_coverage"] * 100))
    print("  Topology changes: %s" % topo_cfg.get("enabled", False))
    print("  Output: %s" % output_path)

    generator = PowerGridDataGeneratorV2(
        system_name=sys_cfg["name"],
        num_scenarios=num_scenarios,
        time_steps=data_cfg["time_steps"],
        pmu_coverage=sys_cfg["pmu_coverage"],
        scada_coverage=sys_cfg["scada_coverage"],
        noise_std_pmu=meas_cfg["pmu"]["noise_std"],
        noise_std_scada=meas_cfg["scada"]["noise_std"],
        parameter_drift_enabled=data_cfg["parameter_drift"].get("enabled", True),
        pv_penetration=data_cfg.get("pv_penetration", 0.35),
        topology_change_enabled=topo_cfg.get("enabled", False),
        seed=args.seed,
    )

    dataset = generator.generate_dataset(
        save_path=output_path,
        topology_mix=topo_cfg.get("enabled", False),
    )

    print("\nDone. %d scenarios saved to %s" % (len(dataset), output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-rate DSSE dataset")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--output", type=str, default=None, help="Output pkl path")
    parser.add_argument(
        "--num_scenarios", type=int, default=None, help="Override num scenarios"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
