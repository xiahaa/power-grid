"""
Staged Multi-Rate Mamba Trainer for DSSE

Implements the 3-stage training strategy:
  Stage 1: Train SCADA stream (freeze PMU stream) - learn spatial + slow temporal
  Stage 2: Train PMU stream (freeze SCADA stream) - learn fast temporal dynamics
  Stage 3: Joint fine-tuning (all parameters) - cross-attention fusion + physics

Supports:
  - Multi-rate data loading with separate SCADA/PMU streams
  - Hierarchical physics constraints (bus/feeder/substation)
  - Topology change detection auxiliary loss
  - Gradient clipping, mixed precision, early stopping
  - TensorBoard and SwanLab logging
  - Checkpoint save/load with best-model tracking
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import swanlab

    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False


class StagedTrainer:
    """
    3-stage trainer for MultiRateMambaFusion.

    Stage 1: SCADA stream only (freeze PMU + fusion)
    Stage 2: PMU stream only (freeze SCADA + fusion)
    Stage 3: Full joint fine-tuning
    """

    def __init__(
        self,
        model: nn.Module,
        physics_constraints: nn.Module,
        topology_detector: Optional[nn.Module] = None,
        config: dict = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.physics_constraints = physics_constraints.to(device)
        self.topology_detector = topology_detector
        if self.topology_detector is not None:
            self.topology_detector = self.topology_detector.to(device)

        self.config = config or {}
        self.device = device

        self.train_losses_history: List[Dict[str, float]] = []
        self.val_losses_history: List[Dict[str, float]] = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.current_stage = 0

    def _freeze_module(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def _unfreeze_module(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = True

    def _get_param_group(self, name: str) -> Optional[nn.Module]:
        model = self.model
        mapping = {
            "spatial_convs": getattr(model, "spatial_convs", None),
            "input_proj": getattr(model, "input_proj", None),
            "pmu_temporal": getattr(model, "pmu_temporal", None),
            "scada_temporal": getattr(model, "scada_temporal", None),
            "fusion": getattr(model, "fusion", None),
            "state_head": getattr(model, "state_head", None),
            "parameter_head": getattr(model, "parameter_head", None),
        }
        return mapping.get(name)

    def _set_stage(self, stage: int) -> Dict[str, List[str]]:
        """
        Configure parameter freezing for a training stage.

        Returns dict mapping 'frozen'/'trainable' to lists of module names.
        """
        self.current_stage = stage
        status = {"frozen": [], "trainable": []}

        if stage == 1:
            freeze_names = ["pmu_temporal", "fusion"]
            for name in freeze_names:
                mod = self._get_param_group(name)
                if mod is not None:
                    self._freeze_module(mod)
                    status["frozen"].append(name)

            all_params = [
                n for n, p in self.model.named_parameters() if p.requires_grad
            ]
            status["trainable"] = all_params

        elif stage == 2:
            freeze_names = ["scada_temporal"]
            for name in freeze_names:
                mod = self._get_param_group(name)
                if mod is not None:
                    self._freeze_module(mod)
                    status["frozen"].append(name)

            self._unfreeze_module(self.model)
            for name in freeze_names:
                mod = self._get_param_group(name)
                if mod is not None:
                    self._freeze_module(mod)

            all_params = [
                n for n, p in self.model.named_parameters() if p.requires_grad
            ]
            status["trainable"] = all_params

        elif stage == 3:
            self._unfreeze_module(self.model)
            all_params = [
                n for n, p in self.model.named_parameters() if p.requires_grad
            ]
            status["trainable"] = all_params

        return status

    def _create_optimizer(self, stage: int) -> torch.optim.Optimizer:
        lr = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 1e-5)

        if stage == 1:
            lr = lr * 0.5
        elif stage == 2:
            lr = lr * 0.8

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if self.topology_detector is not None:
            trainable += list(self.topology_detector.parameters())

        return torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)

    def _create_scheduler(self, optimizer, num_epochs: int):
        scheduler_type = self.config.get("lr_scheduler_type", "cosine")
        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=1e-7
            )
        elif scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    def _merge_measurements(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Merge SCADA and PMU measurements into a single measurements dict
        for the model forward pass."""
        scada = batch["scada_meas"]
        pmu = batch["pmu_meas"]

        seq_len = scada["v_mag"].shape[1]
        num_nodes = scada["v_mag"].shape[2]

        v_mag = scada["v_mag"].clone()
        pmu_v_mag = pmu.get("v_mag", None)
        if pmu_v_mag is not None:
            pmu_mask = batch["pmu_mask"].float()
            v_mag = v_mag * (1.0 - pmu_mask) + pmu_v_mag * pmu_mask

        measurements = {
            "v_mag": v_mag,
            "p_bus": scada["p_bus"],
            "q_bus": scada["q_bus"],
            "v_ang": pmu.get("v_ang", torch.zeros_like(v_mag)),
        }

        for key in measurements:
            measurements[key] = torch.nan_to_num(measurements[key], nan=0.0)

        obs_mask = torch.any(batch["scada_mask"], dim=1)
        pmu_any = torch.any(batch["pmu_mask"], dim=1)
        combined_mask = obs_mask | pmu_any
        if combined_mask.dim() < measurements["v_mag"].dim():
            combined_mask = combined_mask.unsqueeze(1).expand(-1, seq_len, -1)

        return measurements, combined_mask

    def train_epoch(
        self,
        dataloader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        grad_clip: float = 1.0,
        use_amp: bool = False,
    ) -> Dict[str, float]:
        self.model.train()
        if self.topology_detector is not None:
            self.topology_detector.train()

        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        meters = {}
        for key in ["total", "state", "parameter", "physics", "smoothness", "topo_aux"]:
            meters[key] = _AverageMeter()

        pbar = tqdm(dataloader, desc="Stage %d Epoch %d" % (self.current_stage, epoch))

        for batch in pbar:
            edge_index = batch["topology"]["edge_index"].to(self.device)
            edge_attr = batch["topology"]["edge_attr"].to(self.device)
            true_states = {
                k: v.to(self.device) for k, v in batch["true_states"].items()
            }
            true_state_masks = {
                k: v.to(self.device) for k, v in batch["true_state_masks"].items()
            }
            true_params = None
            if getattr(criterion, "use_parameter_loss", True):
                true_params = {
                    k: v.to(self.device) for k, v in batch["parameters"].items()
                }

            measurements, obs_mask = self._merge_measurements(batch)
            measurements = {k: v.to(self.device) for k, v in measurements.items()}
            obs_mask = obs_mask.to(self.device)

            pmu_mask_batch = batch["pmu_mask"].to(self.device)
            scada_mask_batch = batch["scada_mask"].to(self.device)
            has_topo = batch.get("has_topology_change")
            if has_topo is not None:
                has_topo = has_topo.to(self.device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast() if use_amp else _nullcontext():
                pred_states, pred_params = self.model(
                    measurements,
                    edge_index,
                    edge_attr=edge_attr,
                    obs_mask=obs_mask,
                    scada_obs_mask=scada_mask_batch,
                    pmu_obs_mask=pmu_mask_batch,
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

                physics_out = self.physics_constraints(
                    states=pred_states,
                    parameters=pred_params,
                    edge_index=edge_index,
                    measurements=measurements,
                )
                physics_loss = physics_out["total"]

                anomaly_scores = None
                if self.topology_detector is not None and has_topo is not None:
                    anomaly_scores, _ = self.topology_detector(measurements)
                    anomaly_scores = anomaly_scores.mean(dim=1)

                loss, loss_dict = criterion(
                    pred_states=pred_states,
                    true_states=true_states,
                    pred_params=pred_params,
                    true_params=true_params,
                    hierarchical_physics_loss=physics_loss,
                    anomaly_scores=anomaly_scores,
                    has_topology_change=has_topo,
                    pmu_mask=pmu_mask_batch,
                    scada_mask=scada_mask_batch,
                    true_state_masks=true_state_masks,
                )

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=grad_clip,
                )
                optimizer.step()

            batch_size = measurements["v_mag"].shape[0]
            for key in meters:
                if key in loss_dict:
                    meters[key].update(loss_dict[key], batch_size)

            pbar.set_postfix(
                {
                    "loss": "%.4f" % meters["total"].avg,
                    "state": "%.4f" % meters["state"].avg,
                }
            )

        return {k: v.avg for k, v in meters.items()}

    @torch.no_grad()
    def validate(
        self,
        dataloader,
        criterion: nn.Module,
        use_amp: bool = False,
    ) -> Dict[str, float]:
        self.model.eval()
        if self.topology_detector is not None:
            self.topology_detector.eval()

        meters = {}
        for key in ["total", "state", "parameter", "physics", "smoothness", "topo_aux"]:
            meters[key] = _AverageMeter()

        all_pred_states = []
        all_true_states = []
        all_pred_params = []
        all_true_params = []
        use_parameter_loss = getattr(criterion, "use_parameter_loss", True)

        for batch in tqdm(dataloader, desc="Validating"):
            edge_index = batch["topology"]["edge_index"].to(self.device)
            edge_attr = batch["topology"]["edge_attr"].to(self.device)
            true_states = {
                k: v.to(self.device) for k, v in batch["true_states"].items()
            }
            true_state_masks = {
                k: v.to(self.device) for k, v in batch["true_state_masks"].items()
            }
            true_params = None
            if use_parameter_loss:
                true_params = {
                    k: v.to(self.device) for k, v in batch["parameters"].items()
                }

            measurements, obs_mask = self._merge_measurements(batch)
            measurements = {k: v.to(self.device) for k, v in measurements.items()}
            obs_mask = obs_mask.to(self.device)

            pmu_mask_batch = batch["pmu_mask"].to(self.device)
            scada_mask_batch = batch["scada_mask"].to(self.device)
            has_topo = batch.get("has_topology_change")
            if has_topo is not None:
                has_topo = has_topo.to(self.device)

            with torch.cuda.amp.autocast() if use_amp else _nullcontext():
                pred_states, pred_params = self.model(
                    measurements,
                    edge_index,
                    edge_attr=edge_attr,
                    obs_mask=obs_mask,
                    scada_obs_mask=scada_mask_batch,
                    pmu_obs_mask=pmu_mask_batch,
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

                physics_out = self.physics_constraints(
                    states=pred_states,
                    parameters=pred_params,
                    edge_index=edge_index,
                    measurements=measurements,
                )
                physics_loss = physics_out["total"]

                anomaly_scores = None
                if self.topology_detector is not None and has_topo is not None:
                    anomaly_scores, _ = self.topology_detector(measurements)
                    anomaly_scores = anomaly_scores.mean(dim=1)

                loss, loss_dict = criterion(
                    pred_states=pred_states,
                    true_states=true_states,
                    pred_params=pred_params,
                    true_params=true_params,
                    hierarchical_physics_loss=physics_loss,
                    anomaly_scores=anomaly_scores,
                    has_topology_change=has_topo,
                    pmu_mask=pmu_mask_batch,
                    scada_mask=scada_mask_batch,
                    true_state_masks=true_state_masks,
                )

            batch_size = measurements["v_mag"].shape[0]
            for key in meters:
                if key in loss_dict:
                    meters[key].update(loss_dict[key], batch_size)

            masked_pred_states = {}
            masked_true_states = {}
            for key in true_states:
                invalid_fill = torch.full_like(true_states[key], float("nan"))
                masked_true_states[key] = torch.where(
                    true_state_masks[key], true_states[key], invalid_fill
                )
                masked_pred_states[key] = torch.where(
                    true_state_masks[key], pred_states[key], invalid_fill
                )

            all_pred_states.append(masked_pred_states)
            all_true_states.append(masked_true_states)
            if use_parameter_loss:
                all_pred_params.append(pred_params)
                all_true_params.append(true_params)

        metrics = {}
        if len(all_pred_states) > 0:
            pred_cat = {
                k: torch.cat([d[k] for d in all_pred_states])
                for k in all_pred_states[0]
            }
            true_cat = {
                k: torch.cat([d[k] for d in all_true_states])
                for k in all_true_states[0]
            }
            for key in ["v_mag", "v_ang"]:
                if key in pred_cat and key in true_cat:
                    valid = torch.isfinite(pred_cat[key]) & torch.isfinite(true_cat[key])
                    if valid.any():
                        diff = pred_cat[key][valid] - true_cat[key][valid]
                        metrics["%s_rmse" % key] = torch.sqrt((diff**2).mean()).item()
                        metrics["%s_mae" % key] = diff.abs().mean().item()
            try:
                pred_p = {
                    k: torch.cat([d[k] for d in all_pred_params])
                    for k in all_pred_params[0]
                }
                true_p = {
                    k: torch.cat([d[k] for d in all_true_params])
                    for k in all_true_params[0]
                }

                for key in ["r_line", "x_line"]:
                    if key in pred_p and key in true_p:
                        if pred_p[key].shape[-1] != true_p[key].shape[-1]:
                            min_len = min(pred_p[key].shape[-1], true_p[key].shape[-1])
                            pred_p[key] = pred_p[key][..., :min_len]
                            true_p[key] = true_p[key][..., :min_len]
                        metrics["%s_rmse" % key] = torch.sqrt(
                            ((pred_p[key] - true_p[key]) ** 2).mean()
                        ).item()
                        metrics["%s_mae" % key] = (
                            torch.abs(pred_p[key] - true_p[key]).mean().item()
                        )
            except (RuntimeError, IndexError, TypeError):
                pred_p = None
                true_p = None

        results = {**{k: v.avg for k, v in meters.items()}, **metrics}
        return results

    def train_staged(
        self,
        train_loader,
        val_loader,
        criterion: nn.Module,
        staged_config: Dict[str, int],
        save_dir: str,
        grad_clip: float = 1.0,
        use_amp: bool = False,
        writer: Optional[SummaryWriter] = None,
        swanlab_run=None,
        early_stopping_patience: int = 30,
        early_stopping_min_delta: float = 1e-5,
        save_freq: int = 10,
    ) -> Dict:
        """
        Execute full 3-stage training.

        Args:
            staged_config: dict with keys stage1_epochs, stage2_epochs, stage3_epochs
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        stage_epochs = {
            1: staged_config.get("stage1_epochs", 30),
            2: staged_config.get("stage2_epochs", 30),
            3: staged_config.get("stage3_epochs", 140),
        }

        total_epochs = sum(stage_epochs.values())
        global_epoch = 0
        es_patience = early_stopping_patience

        start_time = time.time()

        for stage in [1, 2, 3]:
            num_epochs = stage_epochs[stage]
            stage_best_val_loss = float("inf")
            stage_es_counter = 0
            print("\n" + "=" * 60)
            print("STAGE %d: %d epochs" % (stage, num_epochs))
            print("=" * 60)

            status = self._set_stage(stage)
            print("  Frozen: %s" % (status["frozen"] if status["frozen"] else "None"))
            n_trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
            print("  Trainable params: %d" % n_trainable)

            optimizer = self._create_optimizer(stage)
            scheduler = self._create_scheduler(optimizer, num_epochs)

            for local_epoch in range(1, num_epochs + 1):
                global_epoch += 1

                train_losses = self.train_epoch(
                    train_loader,
                    criterion,
                    optimizer,
                    global_epoch,
                    grad_clip=grad_clip,
                    use_amp=use_amp,
                )

                val_results = self.validate(val_loader, criterion, use_amp=use_amp)

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_results["total"])
                else:
                    scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]

                self.train_losses_history.append(train_losses)
                self.val_losses_history.append(val_results)

                print(
                    "\n  Epoch %d/%d (Stage %d)  LR: %.2e"
                    % (global_epoch, total_epochs, stage, current_lr)
                )
                print("    Train Loss: %.6f" % train_losses["total"])
                print("    Val Loss:   %.6f" % val_results["total"])
                if "v_mag_rmse" in val_results:
                    print("    Val V_mag RMSE: %.6f" % val_results["v_mag_rmse"])
                if "v_ang_rmse" in val_results:
                    print("    Val V_ang RMSE: %.6f" % val_results["v_ang_rmse"])

                if writer is not None:
                    writer.add_scalar("Loss/train", train_losses["total"], global_epoch)
                    writer.add_scalar("Loss/val", val_results["total"], global_epoch)
                    writer.add_scalar("Stage", stage, global_epoch)
                    writer.add_scalar("LR", current_lr, global_epoch)
                    for key, val in val_results.items():
                        if "rmse" in key or "mae" in key or "mape" in key:
                            writer.add_scalar("Metrics/%s" % key, val, global_epoch)

                if swanlab_run is not None:
                    log_dict = {
                        "train/loss_total": train_losses["total"],
                        "train/loss_state": train_losses.get("state", 0),
                        "val/loss_total": val_results["total"],
                        "train/stage": stage,
                        "train/learning_rate": current_lr,
                        "train/global_epoch": global_epoch,
                    }
                    for key, val in val_results.items():
                        if "rmse" in key or "mae" in key or "mape" in key:
                            log_dict["val/%s" % key] = val
                    swanlab_run.log(log_dict, step=global_epoch)

                if (
                    val_results["total"]
                    < stage_best_val_loss - early_stopping_min_delta
                ):
                    stage_best_val_loss = val_results["total"]
                    stage_es_counter = 0
                else:
                    stage_es_counter += 1

                if val_results["total"] < self.best_val_loss - early_stopping_min_delta:
                    self.best_val_loss = val_results["total"]
                    self.best_epoch = global_epoch
                    self._save_checkpoint(
                        optimizer,
                        global_epoch,
                        self.best_val_loss,
                        save_path / "best_model.pt",
                    )
                    print("    ** New best model saved **")

                if global_epoch % save_freq == 0:
                    self._save_checkpoint(
                        optimizer,
                        global_epoch,
                        val_results["total"],
                        save_path / ("checkpoint_epoch_%d.pt" % global_epoch),
                    )

                if stage_es_counter >= es_patience:
                    print(
                        "\n  Early stopping stage %d at epoch %d"
                        % (stage, global_epoch)
                    )
                    break

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training complete in %.2f hours" % (elapsed / 3600))
        print("Best val loss: %.6f at epoch %d" % (self.best_val_loss, self.best_epoch))
        print("=" * 60)

        summary = {
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "total_epochs": global_epoch,
            "training_time_hours": elapsed / 3600,
        }

        if swanlab_run is not None:
            swanlab_run.log({"summary/" + k: v for k, v in summary.items()})
            swanlab_run.finish()

        return summary

    def _save_checkpoint(self, optimizer, epoch, loss, path):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "best_val_loss": self.best_val_loss,
            "current_stage": self.current_stage,
        }
        if self.topology_detector is not None:
            state["topo_detector_state_dict"] = self.topology_detector.state_dict()
        torch.save(state, path)

    def _reconcile_model_state_dict(self, checkpoint_state_dict):
        model_state = self.model.state_dict()
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

    def load_checkpoint(self, path, optimizer=None):
        checkpoint = torch.load(path, map_location=self.device)
        reconciled_state_dict = self._reconcile_model_state_dict(
            checkpoint["model_state_dict"]
        )
        self.model.load_state_dict(reconciled_state_dict)
        if (
            self.topology_detector is not None
            and "topo_detector_state_dict" in checkpoint
        ):
            self.topology_detector.load_state_dict(
                checkpoint["topo_detector_state_dict"]
            )
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.current_stage = checkpoint.get("current_stage", 0)
        print(
            "Loaded checkpoint from epoch %d (val_loss=%.6f)"
            % (checkpoint["epoch"], checkpoint["loss"])
        )
        return checkpoint


class _nullcontext:
    """Fallback for Python < 3.7 contextlib.nullcontext."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
