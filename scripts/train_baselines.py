"""
Training script for baseline DSSE models.

Usage:
    python scripts/train_baselines.py --data data/ieee123_large.pkl --model gnn
    python scripts/train_baselines.py --data data/ieee123_large.pkl --model all
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import time
import json

from utils.utils import set_seed
from data.multi_rate_dataloader import get_multi_rate_dataloader
from models.baselines import (
    WLSEstimator,
    GNNEstimator,
    LSTMEstimator,
    TransformerEstimator,
)


BASELINE_MODELS = {
    "wls": WLSEstimator,
    "gnn": GNNEstimator,
    "lstm": LSTMEstimator,
    "transformer": TransformerEstimator,
}


class BaselineLoss(nn.Module):
    def __init__(self, state_weight=1.0, param_weight=0.0):
        super().__init__()
        self.state_weight = state_weight
        self.param_weight = param_weight

    def forward(self, pred_states, pred_params, true_states, true_params):
        state_loss = 0.0
        for key in ["v_mag", "v_ang"]:
            if key in pred_states and key in true_states:
                state_loss = state_loss + torch.mean(
                    (pred_states[key] - true_states[key]) ** 2
                )

        param_loss = 0.0
        if self.param_weight > 0:
            for key in ["r_line", "x_line"]:
                if key in pred_params and key in true_params:
                    if pred_params[key].shape == true_params[key].shape:
                        param_loss = param_loss + torch.mean(
                            (pred_params[key] - true_params[key]) ** 2
                        )

        return self.state_weight * state_loss + self.param_weight * param_loss


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        scada_meas = batch["scada_meas"]
        pmu_meas = batch["pmu_meas"]
        edge_index = batch["topology"]["edge_index"]
        edge_attr = batch["topology"]["edge_attr"]
        true_states = batch["true_states"]
        true_params = batch["parameters"]

        measurements = {k: v.to(device) for k, v in scada_meas.items()}
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        true_states = {k: v.to(device) for k, v in true_states.items()}
        true_params = {k: v.to(device) for k, v in true_params.items()}

        obs_mask = None

        optimizer.zero_grad()
        pred_states, pred_params = model(measurements, edge_index, edge_attr, obs_mask)
        loss = criterion(pred_states, pred_params, true_states, true_params)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss = total_loss + loss.item()
        num_batches = num_batches + 1

    return total_loss / max(num_batches, 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            scada_meas = batch["scada_meas"]
            pmu_meas = batch["pmu_meas"]
            edge_index = batch["topology"]["edge_index"]
            edge_attr = batch["topology"]["edge_attr"]
            true_states = batch["true_states"]
            true_params = batch["parameters"]

            measurements = {k: v.to(device) for k, v in scada_meas.items()}
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device) if edge_attr is not None else None
            true_states = {k: v.to(device) for k, v in true_states.items()}
            true_params = {k: v.to(device) for k, v in true_params.items()}

            obs_mask = None

            pred_states, pred_params = model(
                measurements, edge_index, edge_attr, obs_mask
            )
            loss = criterion(pred_states, pred_params, true_states, true_params)

            total_loss = total_loss + loss.item()
            num_batches = num_batches + 1

    return total_loss / max(num_batches, 1)


def compute_metrics(model, loader, device):
    model.eval()
    v_mag_errors = []
    v_ang_errors = []

    with torch.no_grad():
        for batch in loader:
            scada_meas = batch["scada_meas"]
            pmu_meas = batch["pmu_meas"]
            edge_index = batch["topology"]["edge_index"]
            edge_attr = batch["topology"]["edge_attr"]
            true_states = batch["true_states"]

            measurements = {k: v.to(device) for k, v in scada_meas.items()}
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device) if edge_attr is not None else None
            true_states = {k: v.to(device) for k, v in true_states.items()}

            obs_mask = None

            pred_states, _ = model(measurements, edge_index, edge_attr, obs_mask)

            v_mag_errors.append(
                torch.mean(torch.abs(pred_states["v_mag"] - true_states["v_mag"]))
                .cpu()
                .item()
            )
            v_ang_errors.append(
                torch.mean(torch.abs(pred_states["v_ang"] - true_states["v_ang"]))
                .cpu()
                .item()
            )

    return {
        "v_mag_mae": sum(v_mag_errors) / len(v_mag_errors),
        "v_ang_mae": sum(v_ang_errors) / len(v_ang_errors),
    }


def train_model(model_name, model_class, data_path, device, epochs=50, lr=0.001):
    print("\n" + "=" * 60)
    print("Training %s baseline" % model_name.upper())
    print("=" * 60)

    set_seed(42)

    train_loader = get_multi_rate_dataloader(
        data_path=data_path,
        batch_size=32,
        split="train",
        num_workers=4,
        sequence_length=10,
    )
    val_loader = get_multi_rate_dataloader(
        data_path=data_path,
        batch_size=32,
        split="val",
        num_workers=4,
        sequence_length=10,
    )

    sample = train_loader.dataset[0]
    num_nodes = sample["true_states"]["v_mag"].shape[0]
    num_lines = sample["parameters"]["r_line"].shape[0]
    print("Num nodes: %d, Num lines: %d" % (num_nodes, num_lines))

    if model_name == "transformer":
        model = model_class(
            num_nodes=num_nodes, num_edges=num_lines, input_dim=3, d_model=128
        )
    else:
        model = model_class(
            num_nodes=num_nodes, num_edges=num_lines, input_dim=3, hidden_dim=128
        )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameters: %d" % n_params)

    criterion = BaselineLoss(state_weight=1.0, param_weight=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    save_dir = Path("checkpoints/baselines")
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                save_dir / ("%s_best.pt" % model_name),
            )
        else:
            patience_counter = patience_counter + 1

        if (epoch + 1) % 5 == 0:
            print(
                "Epoch %d/%d - Train Loss: %.6f, Val Loss: %.6f"
                % (epoch + 1, epochs, train_loss, val_loss)
            )

        if patience_counter >= patience:
            print("Early stopping at epoch %d" % (epoch + 1))
            break

    checkpoint = torch.load(save_dir / ("%s_best.pt" % model_name))
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = compute_metrics(model, val_loader, device)

    results = {
        "model": model_name,
        "best_val_loss": best_val_loss,
        "best_epoch": checkpoint["epoch"],
        "num_params": n_params,
        "metrics": metrics,
    }

    print("\nResults:")
    print("  Best val loss: %.6f (epoch %d)" % (best_val_loss, checkpoint["epoch"]))
    print("  V_mag MAE: %.6f" % metrics["v_mag_mae"])
    print("  V_ang MAE: %.6f" % metrics["v_ang_mae"])

    with open(save_dir / ("%s_results.json" % model_name), "w") as f:
        json.dump(results, f, indent=2)

    return results


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: %s" % device)

    if not os.path.exists(args.data):
        print("ERROR: Data file not found: %s" % args.data)
        sys.exit(1)

    if args.model == "all":
        all_results = {}
        for model_name, model_class in BASELINE_MODELS.items():
            results = train_model(
                model_name, model_class, args.data, device, epochs=args.epochs
            )
            all_results[model_name] = results

        print("\n" + "=" * 60)
        print("BASELINE COMPARISON SUMMARY")
        print("=" * 60)
        print("%-15s %12s %12s %12s" % ("Model", "Val Loss", "V_mag MAE", "V_ang MAE"))
        print("-" * 60)
        for model_name, results in all_results.items():
            print(
                "%-15s %12.6f %12.6f %12.6f"
                % (
                    model_name.upper(),
                    results["best_val_loss"],
                    results["metrics"]["v_mag_mae"],
                    results["metrics"]["v_ang_mae"],
                )
            )

        save_dir = Path("checkpoints/baselines")
        with open(save_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
    else:
        if args.model not in BASELINE_MODELS:
            print(
                "ERROR: Unknown model '%s'. Available: %s"
                % (args.model, list(BASELINE_MODELS.keys()))
            )
            sys.exit(1)

        train_model(
            args.model,
            BASELINE_MODELS[args.model],
            args.data,
            device,
            epochs=args.epochs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline DSSE models")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset pkl")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all"] + list(BASELINE_MODELS.keys()),
        help="Model to train",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    args = parser.parse_args()
    main(args)
