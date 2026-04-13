"""
Training script for Multi-Rate Mamba Fusion DSSE (Phase 2)

Usage:
    python scripts/train_v2.py --config configs/ieee123_config.yaml
    python scripts/train_v2.py --config configs/ieee123_config.yaml --data data/ieee123_dataset.pkl
    python scripts/train_v2.py --config configs/ieee123_config.yaml --stage 3 --resume checkpoints/ieee123/best_model.pt
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import time

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import swanlab

    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

from utils.utils import load_config, resolve_dataset_path, set_seed
from data.multi_rate_dataloader import get_multi_rate_dataloader
from data.ieee123_network import IEEE123_FEEDER_MAP
from models.multi_rate_mamba import MultiRateMambaFusion
from models.topology_adaptive import TopologyChangeDetector
from physics.hierarchical_constraints import (
    HierarchicalPhysicsConstraints,
    NoOpPhysicsConstraints,
)
from train.loss_v2 import MultiRateEstimationLoss
from train.trainer_v2 import StagedTrainer


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


def build_physics(config):
    physics_cfg = config.get("physics", {})
    if not physics_cfg.get("enabled", True):
        return NoOpPhysicsConstraints()

    hier_cfg = physics_cfg.get("hierarchical", {})
    pf_cfg = physics_cfg.get("power_flow", {})

    return HierarchicalPhysicsConstraints(
        num_feeders=config["system"].get("num_feeders", 6),
        bus_weight=hier_cfg.get("bus_weight", 1.0),
        feeder_weight=hier_cfg.get("feeder_weight", 0.5),
        substation_weight=hier_cfg.get("substation_weight", 0.2),
        voltage_limits=tuple(pf_cfg.get("voltage_limits", [0.95, 1.05])),
        constraint_type=physics_cfg.get("constraint_type", "soft"),
        projection_method=physics_cfg.get("projection_method", "gradient_descent"),
        max_iterations=physics_cfg.get("max_iterations", 20),
        tolerance=physics_cfg.get("tolerance", 1e-4),
        power_balance_weight=pf_cfg.get("power_balance_weight", 10.0),
        slack_bus=pf_cfg.get("slack_bus"),
        hierarchical_enabled=hier_cfg.get("enabled", True),
    )


def build_topology_detector(config):
    topo_cfg = config["model"].get("topology_detector", {})
    return TopologyChangeDetector(
        input_dim=5,
        hidden_dim=topo_cfg.get("hidden_dim", 32),
        threshold=topo_cfg.get("threshold", 0.7),
        window_size=topo_cfg.get("window_size", 10),
    )


def build_loss(config):
    loss_cfg = config.get("loss", {})
    task_cfg = config.get("task", {})
    feeder_map = None
    if config.get("system", {}).get("name") == "ieee123":
        feeder_map = IEEE123_FEEDER_MAP
    estimate_line_parameters = task_cfg.get(
        "estimate_line_parameters", loss_cfg.get("parameter_weight", 0.5) > 0
    )
    return MultiRateEstimationLoss(
        state_weight=loss_cfg.get("state_weight", 1.0),
        parameter_weight=(
            loss_cfg.get("parameter_weight", 0.5) if estimate_line_parameters else 0.0
        ),
        physics_weight=loss_cfg.get("physics_weight", 0.0001),
        temporal_smoothness=loss_cfg.get("temporal_smoothness", 0.01),
        pmu_loss_weight=loss_cfg.get("pmu_loss_weight", 1.0),
        scada_loss_weight=loss_cfg.get("scada_loss_weight", 0.5),
        feeder_map=feeder_map,
        feeder_loss_weights=loss_cfg.get("feeder_loss_weights"),
    )


def get_split_max_sequences(config, split: str):
    max_cfg = config.get("data", {}).get("max_sequences")
    if isinstance(max_cfg, dict):
        return max_cfg.get(split)
    return max_cfg


def main(args):
    config = load_config(args.config)
    set_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        and config["hardware"].get("device", "cuda") == "cuda"
        else "cpu"
    )
    print("Device: %s" % device)
    print(
        "System: %s (%d buses)"
        % (config["system"]["name"], config["system"]["num_buses"])
    )

    if args.data:
        data_path = resolve_dataset_path(config, args.data)
    else:
        data_path = resolve_dataset_path(config)

    if not os.path.exists(data_path):
        print("ERROR: Data file not found: %s" % data_path)
        print("Checked default path, config override, and available dataset fallbacks.")
        print("Generate data first with: python scripts/generate_data_v2.py --config %s" % args.config)
        sys.exit(1)

    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    sequence_length = config.get("data", {}).get("sequence_length", 10)
    prediction_horizon = config.get("data", {}).get("prediction_horizon", 1)

    print("Loading data from %s ..." % data_path)
    train_loader = get_multi_rate_dataloader(
        data_path=data_path,
        batch_size=batch_size,
        split="train",
        num_workers=config["hardware"].get("num_workers", 4),
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        max_sequences=get_split_max_sequences(config, "train"),
    )
    val_loader = get_multi_rate_dataloader(
        data_path=data_path,
        batch_size=batch_size,
        split="val",
        num_workers=config["hardware"].get("num_workers", 4),
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        max_sequences=get_split_max_sequences(config, "val"),
    )
    print("Train batches: %d, Val batches: %d" % (len(train_loader), len(val_loader)))

    sample = train_loader.dataset[0]
    num_nodes = sample["scada_meas"]["v_mag"].shape[-1]
    num_edges = sample["topology"]["edge_index"].shape[1]
    config_num_buses = config["system"].get("num_buses")
    if config_num_buses is not None and config_num_buses != num_nodes:
        print(
            "WARNING: config num_buses=%d but dataset provides %d nodes"
            % (config_num_buses, num_nodes)
        )
    print("Num edges: %d" % num_edges)

    print("\nBuilding model...")
    model = build_model(config, num_nodes, num_edges)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameters: %d" % n_params)

    physics = build_physics(config)
    topo_detector = build_topology_detector(config)

    criterion = build_loss(config)

    save_dir = Path(
        train_cfg.get("save_dir", "checkpoints/%s" % config["system"]["name"])
    )

    writer = None
    if TENSORBOARD_AVAILABLE and config["logging"].get("use_tensorboard", True):
        log_dir = Path(
            config["logging"].get("log_dir", "logs/%s" % config["system"]["name"])
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print("TensorBoard logging: %s" % log_dir)

    swanlab_run = None
    if config["logging"].get("use_swanlab", False) and SWANLAB_AVAILABLE:
        swanlab_run = swanlab.init(
            project="power-grid-dsse",
            experiment_name="mrf-mamba-%s" % config["system"]["name"],
            config=config,
        )
        print("SwanLab logging enabled")

    trainer = StagedTrainer(
        model=model,
        physics_constraints=physics,
        topology_detector=topo_detector,
        config={
            "learning_rate": train_cfg["learning_rate"],
            "weight_decay": train_cfg.get("weight_decay", 1e-5),
            "lr_scheduler_type": train_cfg["lr_scheduler"].get("type", "cosine"),
        },
        device=device,
    )

    if args.resume:
        print("Resuming from %s" % args.resume)
        trainer.load_checkpoint(args.resume)

    staged_cfg = train_cfg.get("staged_training", {})
    es_cfg = train_cfg.get("early_stopping", {})

    summary = trainer.train_staged(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        staged_config=staged_cfg,
        save_dir=str(save_dir),
        grad_clip=train_cfg.get("grad_clip", 1.0),
        use_amp=args.amp,
        writer=writer,
        swanlab_run=swanlab_run,
        early_stopping_patience=es_cfg.get("patience", 30),
        early_stopping_min_delta=es_cfg.get("min_delta", 1e-5),
        save_freq=train_cfg.get("save_freq", 10),
    )

    print("\nFinal results:")
    for k, v in summary.items():
        print("  %s: %s" % (k, v))

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Rate Mamba Fusion DSSE")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset pkl")
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint path to resume"
    )
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    args = parser.parse_args()
    main(args)
