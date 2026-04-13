"""
Evaluation script for Multi-Rate Mamba Fusion DSSE

Usage:
    python scripts/evaluate_v2.py --config configs/ieee123_config.yaml --checkpoint checkpoints/ieee123/best_model.pt
    python scripts/evaluate_v2.py --config configs/ieee123_config.yaml --checkpoint checkpoints/ieee123/best_model.pt --split test
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm

from utils.utils import load_config, resolve_dataset_path
from data.multi_rate_dataloader import get_multi_rate_dataloader
from data.ieee123_network import IEEE123_FEEDER_MAP
from models.multi_rate_mamba import MultiRateMambaFusion
from models.topology_adaptive import TopologyChangeDetector
from physics.hierarchical_constraints import (
    HierarchicalPhysicsConstraints,
    NoOpPhysicsConstraints,
)
from evaluation.metrics import DSSEEvaluator
from train.loss_v2 import MultiRateEstimationLoss


def build_model(config, num_nodes, num_edges):
    model_cfg = config["model"]
    spatial = model_cfg["spatial_encoder"]
    pmu = model_cfg["pmu_stream"]
    scada = model_cfg["scada_stream"]
    fusion = model_cfg["fusion"]
    state_head = model_cfg["state_head"]
    param_head = model_cfg["parameter_head"]
    feeder_refine_cfg = state_head.get("feeder_refinement", {})
    feeder_map = None
    if feeder_refine_cfg.get("enabled", False) and config["system"].get("name") == "ieee123":
        feeder_map = IEEE123_FEEDER_MAP

    return MultiRateMambaFusion(
        num_nodes=num_nodes,
        num_edges=num_edges,
        input_dim=4,
        spatial_hidden_dim=spatial["hidden_dim"],
        spatial_num_heads=spatial["num_heads"],
        spatial_num_layers=spatial["num_layers"],
        pmu_d_model=pmu["d_model"],
        pmu_d_state=pmu["d_state"],
        pmu_num_layers=pmu["num_layers"],
        scada_d_model=scada["d_model"],
        scada_d_state=scada["d_state"],
        scada_num_layers=scada["num_layers"],
        fusion_dim=fusion["fusion_dim"],
        fusion_num_heads=fusion["num_heads"],
        state_hidden_dims=state_head["hidden_dims"],
        parameter_hidden_dims=param_head["hidden_dims"],
        feeder_map=feeder_map,
        feeder_emb_dim=feeder_refine_cfg.get("embedding_dim", 0),
        feeder_refine_hidden_dim=feeder_refine_cfg.get("hidden_dim", 32),
        measurement_refine_dim=feeder_refine_cfg.get("measurement_feature_dim", 0),
        feeder_target_ids=feeder_refine_cfg.get("target_feeders"),
        feeder_refine_scale=feeder_refine_cfg.get("residual_scale", 1.0),
    )


def reconcile_model_state_dict(model, checkpoint_state_dict):
    model_state = model.state_dict()
    reconciled = {k: v.clone() for k, v in model_state.items()}

    for key, value in checkpoint_state_dict.items():
        if key in reconciled and reconciled[key].shape == value.shape:
            reconciled[key] = value

    proj_key = "input_proj.weight"
    if proj_key in reconciled and proj_key in model_state:
        ckpt_weight = checkpoint_state_dict.get(proj_key, reconciled[proj_key])
        model_weight = model_state[proj_key]
        if ckpt_weight.shape != model_weight.shape:
            adapted = model_weight.clone()
            rows = min(ckpt_weight.shape[0], adapted.shape[0])
            cols = min(ckpt_weight.shape[1], adapted.shape[1])
            adapted[:rows, :cols] = ckpt_weight[:rows, :cols]
            if adapted.shape[1] > ckpt_weight.shape[1]:
                adapted[:, ckpt_weight.shape[1] :] = 0.0
            reconciled[proj_key] = adapted

    return reconciled


def evaluate_model(
    model,
    dataloader,
    evaluator,
    device,
    criterion=None,
    has_topology_detector=False,
    topo_detector=None,
    model_num_edges=258,
):
    model.eval()
    if topo_detector is not None:
        topo_detector.eval()

    all_pred_states = []
    all_true_states = []
    all_pred_params = []
    all_true_params = []

    criterion = criterion or MultiRateEstimationLoss()
    use_parameter_loss = getattr(criterion, "use_parameter_loss", True)

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        edge_index = batch["topology"]["edge_index"].to(device)
        edge_attr = batch["topology"]["edge_attr"].to(device)
        true_states = {k: v.to(device) for k, v in batch["true_states"].items()}
        true_state_masks = {
            k: v.to(device) for k, v in batch["true_state_masks"].items()
        }
        true_params = None
        if use_parameter_loss:
            true_params = {k: v.to(device) for k, v in batch["parameters"].items()}

        scada = batch["scada_meas"]
        pmu = batch["pmu_meas"]

        v_mag = scada["v_mag"].to(device)
        pmu_v = pmu.get("v_mag", None)
        if pmu_v is not None:
            pmu_mask = batch["pmu_mask"].float().to(device)
            v_mag = v_mag * (1.0 - pmu_mask) + pmu_v.to(device) * pmu_mask

        measurements = {
            "v_mag": torch.nan_to_num(v_mag, nan=0.0),
            "p_bus": torch.nan_to_num(scada["p_bus"].to(device), nan=0.0),
            "q_bus": torch.nan_to_num(scada["q_bus"].to(device), nan=0.0),
            "v_ang": torch.nan_to_num(pmu.get("v_ang", torch.zeros_like(v_mag)).to(device), nan=0.0),
        }

        with torch.no_grad():
            num_edges_batch = edge_index.shape[1]
            if num_edges_batch != model_num_edges:
                continue
            pred_states, pred_params = model(
                measurements,
                edge_index,
                edge_attr=edge_attr,
                obs_mask=(torch.any(batch["scada_mask"], dim=1).to(device) | torch.any(batch["pmu_mask"], dim=1).to(device)).unsqueeze(1).expand(-1, measurements["v_mag"].shape[1], -1),
                scada_obs_mask=batch["scada_mask"].to(device),
                pmu_obs_mask=batch["pmu_mask"].to(device),
            )

            if (
                true_params is not None
                and isinstance(true_params["r_line"], torch.Tensor)
                and true_params["r_line"].dim() == 1
            ):
                batch_size = pred_params["r_line"].shape[0]
                true_params["r_line"] = (
                    true_params["r_line"].unsqueeze(0).expand(batch_size, -1)
                )
                true_params["x_line"] = (
                    true_params["x_line"].unsqueeze(0).expand(batch_size, -1)
                )

            if criterion.physics_weight > 0.0:
                physics = HierarchicalPhysicsConstraints(num_feeders=6).to(device)
                physics_out = physics(
                    states={k: v.cpu() for k, v in pred_states.items()},
                    parameters={k: v.cpu() for k, v in pred_params.items()},
                    edge_index=edge_index.cpu(),
                    measurements={k: v.cpu() for k, v in measurements.items()},
                )
                physics_total = physics_out["total"]
            else:
                physics_total = torch.tensor(0.0)

            loss, _ = criterion(
                {k: v.cpu() for k, v in pred_states.items()},
                {k: v.cpu() for k, v in true_states.items()},
                ({k: v.cpu() for k, v in pred_params.items()} if use_parameter_loss else None),
                ({k: v.cpu() for k, v in true_params.items()} if use_parameter_loss else None),
                physics_total,
                true_state_masks={k: v.cpu() for k, v in true_state_masks.items()},
            )
            total_loss += loss.item()
            n_batches += 1

        masked_pred_states = {}
        masked_true_states = {}
        for key in true_states:
            invalid_fill = torch.full_like(true_states[key], float("nan"))
            masked_true_states[key] = torch.where(
                true_state_masks[key], true_states[key], invalid_fill
            ).cpu()
            masked_pred_states[key] = torch.where(
                true_state_masks[key], pred_states[key], invalid_fill
            ).cpu()

        all_pred_states.append(masked_pred_states)
        all_true_states.append(masked_true_states)
        if use_parameter_loss:
            all_pred_params.append({k: v.cpu() for k, v in pred_params.items()})
            all_true_params.append({k: v.cpu() for k, v in true_params.items()})

    results = evaluator.evaluate_batch(
        all_pred_states,
        all_true_states,
        (all_pred_params if use_parameter_loss else None),
        (all_true_params if use_parameter_loss else None),
    )
    results["avg_loss"] = total_loss / max(1, n_batches)

    return results


def get_split_max_sequences(config, split: str):
    max_cfg = config.get("data", {}).get("max_sequences")
    if isinstance(max_cfg, dict):
        return max_cfg.get(split)
    return max_cfg


def main(args):
    config = load_config(args.config)
    device = (
        "cuda"
        if torch.cuda.is_available()
        and config["hardware"].get("device", "cuda") == "cuda"
        else "cpu"
    )
    print("Device: %s" % device)

    if args.data:
        data_path = resolve_dataset_path(config, args.data)
    else:
        data_path = resolve_dataset_path(config)

    if not os.path.exists(data_path):
        print("ERROR: Data file not found: %s" % data_path)
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print("ERROR: Checkpoint not found: %s" % args.checkpoint)
        sys.exit(1)

    loss_cfg = config.get("loss", {})
    task_cfg = config.get("task", {})
    estimate_line_parameters = task_cfg.get(
        "estimate_line_parameters", loss_cfg.get("parameter_weight", 0.5) > 0
    )

    batch_size = config["training"]["batch_size"]
    split = args.split
    sequence_length = config.get("data", {}).get("sequence_length", 10)
    prediction_horizon = config.get("data", {}).get("prediction_horizon", 1)

    loader = get_multi_rate_dataloader(
        data_path=data_path,
        batch_size=batch_size,
        split=split,
        num_workers=config["hardware"].get("num_workers", 4),
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        max_sequences=get_split_max_sequences(config, split),
    )

    sample = loader.dataset[0]
    num_nodes = sample["scada_meas"]["v_mag"].shape[-1]
    num_edges = sample["topology"]["edge_index"].shape[1]
    config_num_buses = config["system"].get("num_buses")
    if config_num_buses is not None and config_num_buses != num_nodes:
        print(
            "WARNING: config num_buses=%d but dataset provides %d nodes"
            % (config_num_buses, num_nodes)
        )

    print("Building model (num_edges=%d) ..." % num_edges)
    model = build_model(config, num_nodes, num_edges)

    print("Loading checkpoint: %s" % args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(reconcile_model_state_dict(model, checkpoint["model_state_dict"]))
    model = model.to(device)

    topo_detector = None
    if args.topo_checkpoint and os.path.exists(args.topo_checkpoint):
        topo_cfg = config["model"].get("topology_detector", {})
        topo_detector = TopologyChangeDetector(
            input_dim=5,
            hidden_dim=topo_cfg.get("hidden_dim", 32),
            threshold=topo_cfg.get("threshold", 0.7),
            window_size=topo_cfg.get("window_size", 10),
        ).to(device)
        if "topo_detector_state_dict" in checkpoint:
            topo_detector.load_state_dict(checkpoint["topo_detector_state_dict"])
            print("Topology detector loaded")

    pf_cfg = config.get("physics", {}).get("power_flow", {})
    v_limits = tuple(pf_cfg.get("voltage_limits", [0.95, 1.05]))

    feeder_map = None
    try:
        from data.ieee123_network import get_feeder_map

        feeder_map = get_feeder_map()
    except Exception:
        pass

    evaluator = DSSEEvaluator(
        feeder_map=feeder_map,
        v_min=v_limits[0],
        v_max=v_limits[1],
        include_parameters=estimate_line_parameters,
    )

    criterion = MultiRateEstimationLoss(
        state_weight=loss_cfg.get("state_weight", 1.0),
        parameter_weight=(
            loss_cfg.get("parameter_weight", 0.5) if estimate_line_parameters else 0.0
        ),
        physics_weight=loss_cfg.get("physics_weight", 0.0001),
        temporal_smoothness=loss_cfg.get("temporal_smoothness", 0.01),
        pmu_loss_weight=loss_cfg.get("pmu_loss_weight", 1.0),
        scada_loss_weight=loss_cfg.get("scada_loss_weight", 0.5),
    )

    print("\nEvaluating on %s split (%d batches) ..." % (split, len(loader)))
    results = evaluate_model(
        model,
        loader,
        evaluator,
        device,
        criterion=criterion,
        topo_detector=topo_detector,
        model_num_edges=num_edges,
    )

    print("\n" + evaluator.format_results(results))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {}
        for k, v in results.items():
            if isinstance(v, (int, float, str, bool)):
                serializable[k] = v
            elif isinstance(v, list):
                if len(v) > 0 and isinstance(v[0], (int, float)):
                    serializable[k] = v
            elif isinstance(v, dict):
                inner = {}
                for ik, iv in v.items():
                    if isinstance(iv, (int, float)):
                        inner[str(ik)] = iv
                if inner:
                    serializable[k] = inner

        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print("Results saved to %s" % args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Multi-Rate Mamba Fusion DSSE"
    )
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint path"
    )
    parser.add_argument(
        "--topo-checkpoint", type=str, default=None, help="Topology detector checkpoint"
    )
    parser.add_argument("--data", type=str, default=None, help="Dataset pkl path")
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON path for results"
    )
    args = parser.parse_args()
    main(args)
