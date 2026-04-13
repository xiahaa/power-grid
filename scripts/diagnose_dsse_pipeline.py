"""
Diagnose active DSSE training pathways before launching new experiments.

This script inspects:
1. Config-to-runtime mismatches for physics/loss settings
2. Which physics terms are actually active from model outputs
3. Whether topology features are informative under current trainer wiring
4. Feeder-wise PMU/SCADA observation coverage on a sample batch
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import json
from pathlib import Path

import torch

from utils.utils import load_config, resolve_dataset_path, set_seed
from utils.dsse_diagnostics import (
    inspect_loss_configuration,
    inspect_physics_config_usage,
    inspect_state_support_for_physics,
    inspect_topology_feature_tensor,
    summarize_mask_by_feeder,
)
from data.ieee123_network import IEEE123_FEEDER_MAP
from data.multi_rate_dataloader import get_multi_rate_dataloader
from physics.hierarchical_constraints import prepare_states_for_physics
from train.trainer_v2 import StagedTrainer
from train_v2 import build_model, build_physics, build_topology_detector


def tensor_to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def main(args):
    set_seed(42)
    config = load_config(args.config)

    if args.data:
        data_path = resolve_dataset_path(config, args.data)
    else:
        data_path = resolve_dataset_path(config)

    device = (
        "cuda"
        if torch.cuda.is_available()
        and config.get("hardware", {}).get("device", "cuda") == "cuda"
        else "cpu"
    )

    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    loader = get_multi_rate_dataloader(
        data_path=data_path,
        batch_size=args.batch_size,
        split=args.split,
        num_workers=0,
        sequence_length=data_cfg.get("sequence_length", 10),
        prediction_horizon=data_cfg.get("prediction_horizon", 1),
        max_sequences=args.max_sequences,
    )

    batch = next(iter(loader))
    num_nodes = batch["scada_meas"]["v_mag"].shape[-1]
    num_edges = batch["topology"]["edge_index"].shape[1]

    model = build_model(config, num_nodes, num_edges)
    physics = build_physics(config)
    topology_detector = build_topology_detector(config)
    trainer = StagedTrainer(
        model=model,
        physics_constraints=physics,
        topology_detector=topology_detector,
        config={
            "learning_rate": train_cfg.get("learning_rate", 0.001),
            "weight_decay": train_cfg.get("weight_decay", 1e-5),
            "lr_scheduler_type": train_cfg.get("lr_scheduler", {}).get(
                "type", "cosine"
            ),
        },
        device=device,
    )

    checkpoint_status = {
        "path": args.checkpoint,
        "loaded": False,
        "error": None,
    }
    if args.checkpoint:
        try:
            trainer.load_checkpoint(args.checkpoint)
            checkpoint_status["loaded"] = True
        except RuntimeError as exc:
            checkpoint_status["error"] = str(exc)

    measurements, obs_mask = trainer._merge_measurements(batch)
    measurements = {k: v.to(device) for k, v in measurements.items()}
    obs_mask = obs_mask.to(device)
    edge_index = batch["topology"]["edge_index"].to(device)
    edge_attr = batch["topology"]["edge_attr"].to(device)

    trainer.model.eval()
    with torch.no_grad():
        pred_states, pred_params = trainer.model(
            measurements,
            edge_index,
            edge_attr=edge_attr,
            obs_mask=obs_mask,
        )
        physics_states = prepare_states_for_physics(pred_states, measurements)
        physics_out = physics(
            states=pred_states,
            parameters=pred_params,
            edge_index=edge_index,
            measurements=measurements,
        )
        topo_features = topology_detector.compute_features(measurements)
        topo_scores, affected_nodes = topology_detector(measurements)

    report = {
        "config": {
            "config_path": args.config,
            "data_path": data_path,
            "split": args.split,
            "device": device,
            "batch_size": args.batch_size,
            "max_sequences": args.max_sequences,
            "checkpoint": checkpoint_status,
        },
        "dataset": {
            "num_sequences": len(loader.dataset),
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "topology_change_fraction": float(
                torch.stack(
                    [loader.dataset[idx]["has_topology_change"] for idx in range(len(loader.dataset))]
                )
                .float()
                .mean()
                .item()
            )
            if len(loader.dataset) > 0
            else 0.0,
        },
        "config_runtime": {
            "physics": inspect_physics_config_usage(config),
            "loss": inspect_loss_configuration(config),
        },
        "model_forward": {
            "pred_state_shapes": {
                key: list(value.shape) for key, value in pred_states.items()
            },
            "pred_parameter_shapes": {
                key: list(value.shape) for key, value in pred_params.items()
            },
        },
        "physics_runtime": {
            "state_support": inspect_state_support_for_physics(physics_states),
            "loss_components": {
                key: tensor_to_float(value) for key, value in physics_out.items()
            },
        },
        "topology_runtime": {
            "feature_stats": inspect_topology_feature_tensor(topo_features.cpu()),
            "mean_anomaly_score": float(topo_scores.mean().item()),
            "num_affected_nodes": len(affected_nodes),
            "affected_nodes_preview": affected_nodes[:10],
        },
        "measurement_coverage": {
            "pmu_by_feeder": summarize_mask_by_feeder(
                batch["pmu_mask"], IEEE123_FEEDER_MAP
            ),
            "scada_by_feeder": summarize_mask_by_feeder(
                batch["scada_mask"], IEEE123_FEEDER_MAP
            ),
        },
        "notes": [],
    }

    if report["config_runtime"]["physics"]["has_ignored_runtime_mismatch"]:
        report["notes"].append(
            "Physics config contains fields not consumed by build_physics()."
        )
    if checkpoint_status["error"] is not None:
        report["notes"].append(
            "Checkpoint could not be loaded into the current model; report uses randomly initialized weights."
        )
    if not report["physics_runtime"]["state_support"]["feeder_loss_active"]:
        report["notes"].append(
            "Feeder/substation physics terms are inactive because model states lack p_bus/q_bus."
        )
    if report["topology_runtime"]["feature_stats"]["zero_fraction"] >= 0.99:
        report["notes"].append(
            "Topology detector features are effectively all zeros under current trainer wiring."
        )
    if report["config_runtime"]["loss"]["physics_effectively_tiny"]:
        report["notes"].append(
            "Physics weight is tiny, so no-physics ablations are expected to have limited effect."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    print("Saved diagnostic report to %s" % output_path)
    print(json.dumps(report["notes"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sequences", type=int, default=32)
    parser.add_argument(
        "--output",
        default="results/ieee123_dsse_pipeline_diagnosis.json",
    )
    main(parser.parse_args())