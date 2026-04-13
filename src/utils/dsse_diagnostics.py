from typing import Dict, List, Optional

import torch


IGNORED_PHYSICS_CONFIG_PATHS = [
]

USED_PHYSICS_CONFIG_PATHS = [
    "physics.enabled",
    "physics.constraint_type",
    "physics.projection_method",
    "physics.max_iterations",
    "physics.tolerance",
    "physics.hierarchical.enabled",
    "physics.hierarchical.bus_weight",
    "physics.hierarchical.feeder_weight",
    "physics.hierarchical.substation_weight",
    "physics.power_flow.slack_bus",
    "physics.power_flow.voltage_limits",
    "physics.power_flow.power_balance_weight",
    "system.num_feeders",
]


def _nested_get(config: Dict, path: str):
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def inspect_physics_config_usage(config: Dict) -> Dict:
    present_ignored = {
        path: _nested_get(config, path)
        for path in IGNORED_PHYSICS_CONFIG_PATHS
        if _nested_get(config, path) is not None
    }
    present_used = {
        path: _nested_get(config, path)
        for path in USED_PHYSICS_CONFIG_PATHS
        if _nested_get(config, path) is not None
    }
    return {
        "used_paths": present_used,
        "ignored_paths": present_ignored,
        "has_ignored_runtime_mismatch": len(present_ignored) > 0,
    }


def inspect_loss_configuration(config: Dict) -> Dict:
    task_cfg = config.get("task", {})
    loss_cfg = config.get("loss", {})
    estimate_line_parameters = task_cfg.get(
        "estimate_line_parameters", loss_cfg.get("parameter_weight", 0.5) > 0
    )
    parameter_weight = (
        loss_cfg.get("parameter_weight", 0.5) if estimate_line_parameters else 0.0
    )
    return {
        "state_weight": loss_cfg.get("state_weight", 1.0),
        "parameter_weight": parameter_weight,
        "physics_weight": loss_cfg.get("physics_weight", 0.0001),
        "temporal_smoothness": loss_cfg.get("temporal_smoothness", 0.01),
        "topology_aux_weight": loss_cfg.get("topo_aux_weight", 0.1),
        "estimate_line_parameters": estimate_line_parameters,
        "parameter_loss_active": parameter_weight > 0.0,
        "physics_effectively_tiny": loss_cfg.get("physics_weight", 0.0001) <= 1e-4,
    }


def inspect_state_support_for_physics(states: Dict[str, torch.Tensor]) -> Dict:
    missing = [key for key in ["p_bus", "q_bus"] if key not in states]
    return {
        "state_keys": sorted(states.keys()),
        "feeder_loss_active": "p_bus" in states,
        "substation_loss_active": "p_bus" in states and "q_bus" in states,
        "missing_state_keys": missing,
    }


def inspect_topology_feature_tensor(features: torch.Tensor) -> Dict:
    if features.numel() == 0:
        return {
            "shape": list(features.shape),
            "zero_fraction": 1.0,
            "mean_abs_per_channel": [],
            "all_zero_channels": [],
        }

    flat = features.reshape(-1, features.shape[-1])
    mean_abs_per_channel = flat.abs().mean(dim=0)
    return {
        "shape": list(features.shape),
        "zero_fraction": float((features == 0).float().mean().item()),
        "mean_abs_per_channel": [float(x.item()) for x in mean_abs_per_channel],
        "all_zero_channels": [
            idx
            for idx, value in enumerate(mean_abs_per_channel.tolist())
            if abs(value) < 1e-12
        ],
    }


def build_bus_to_feeder_map(feeder_map: Dict[int, List[int]]) -> Dict[int, int]:
    bus_to_feeder = {}
    for feeder_id, buses in feeder_map.items():
        for bus in buses:
            bus_to_feeder[bus] = feeder_id
    return bus_to_feeder


def summarize_mask_by_feeder(
    mask: torch.Tensor,
    feeder_map: Dict[int, List[int]],
    one_indexed_bus_numbers: bool = True,
) -> Dict[str, Dict[str, float]]:
    if mask.dim() != 3:
        raise ValueError("Expected mask with shape [batch, seq, nodes]")

    summary = {}
    num_nodes = mask.shape[-1]
    for feeder_id, buses in feeder_map.items():
        indices = []
        for bus in buses:
            idx = bus - 1 if one_indexed_bus_numbers else bus
            if 0 <= idx < num_nodes:
                indices.append(idx)

        if not indices:
            summary[str(feeder_id)] = {
                "num_buses_in_dataset": 0,
                "observed_fraction": 0.0,
            }
            continue

        feeder_mask = mask[:, :, indices].float()
        summary[str(feeder_id)] = {
            "num_buses_in_dataset": len(indices),
            "observed_fraction": float(feeder_mask.mean().item()),
        }

    return summary