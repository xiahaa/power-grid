"""
Topology-Aware Training Integration for DSSE

Extends StagedTrainer with online topology change handling:
  - Detects topology changes during training via TopologyChangeDetector
  - Resets Mamba hidden states selectively via SelectiveMambaStateReset
  - Triggers IncrementalGATUpdater for subgraph fine-tuning
  - Tracks topology adaptation latency and accuracy metrics
  - Supports curriculum learning: gradually increase topology change frequency
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .trainer_v2 import StagedTrainer, _nullcontext, _AverageMeter


class TopologyAwareTrainer(StagedTrainer):
    """
    Extended trainer that handles topology changes online during training.

    Training flow per batch:
      1. Forward pass through MultiRateMambaFusion
      2. TopologyChangeDetector checks for anomalies
      3. If change detected:
         a. SelectiveMambaStateReset resets affected node states
         b. IncrementalGATUpdater fine-tunes spatial encoder on subgraph
      4. Compute loss (with topology aux loss)
      5. Backward + optimize
    """

    def __init__(
        self,
        model: nn.Module,
        physics_constraints: nn.Module,
        topology_detector: nn.Module,
        state_resetter=None,
        gat_updater=None,
        config: dict = None,
        device: str = "cpu",
    ):
        super().__init__(
            model=model,
            physics_constraints=physics_constraints,
            topology_detector=topology_detector,
            config=config,
            device=device,
        )
        try:
            from ..models.topology_adaptive import SelectiveMambaStateReset
            from ..models.topology_adaptive import IncrementalGATUpdater
        except ImportError:
            from models.topology_adaptive import SelectiveMambaStateReset
            from models.topology_adaptive import IncrementalGATUpdater

        self.state_resetter = state_resetter or SelectiveMambaStateReset(
            reset_mode="partial"
        )
        self.gat_updater = gat_updater
        self.topo_stats = {
            "num_detections": 0,
            "num_true_positives": 0,
            "num_false_positives": 0,
            "num_false_negatives": 0,
            "adaptation_times_ms": [],
            "reset_counts": [],
        }

    def reset_topo_stats(self):
        self.topo_stats = {
            "num_detections": 0,
            "num_true_positives": 0,
            "num_false_positives": 0,
            "num_false_negatives": 0,
            "adaptation_times_ms": [],
            "reset_counts": [],
        }

    def _handle_topology_change(
        self,
        measurements: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        has_topo: bool,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[List[int]], float]:
        """
        Detect and adapt to topology changes.

        Returns:
            affected_nodes: List of affected node indices (or None)
            adaptation_time_ms: Time spent on adaptation in ms
        """
        start = time.time()

        anomaly_scores, affected_nodes = self.topology_detector(measurements)
        self.topo_stats["num_detections"] += 1

        if has_topo and len(affected_nodes) > 0:
            self.topo_stats["num_true_positives"] += 1
        elif not has_topo and len(affected_nodes) > 0:
            self.topo_stats["num_false_positives"] += 1
        elif has_topo and len(affected_nodes) == 0:
            self.topo_stats["num_false_negatives"] += 1

        if len(affected_nodes) > 0:
            if hidden_state is not None:
                hidden_state = self.state_resetter.compute_reset(
                    affected_nodes, hidden_state, edge_index
                )
                self.topo_stats["reset_counts"].append(len(affected_nodes))

            if self.gat_updater is not None and len(affected_nodes) > 0:
                subgraph_nodes = self.gat_updater.get_affected_subgraph(
                    affected_nodes, edge_index
                )
                if (
                    len(subgraph_nodes) > 0
                    and len(subgraph_nodes) < edge_index.max().item() + 1
                ):
                    node_features = measurements["v_mag"]
                    if node_features.dim() > 2:
                        node_features = node_features[:, -1, :]
                    self.gat_updater.incremental_update(
                        affected_nodes, edge_index, node_features, num_steps=10
                    )

        elapsed_ms = (time.time() - start) * 1000
        self.topo_stats["adaptation_times_ms"].append(elapsed_ms)

        return affected_nodes, elapsed_ms

    def train_epoch_with_topology(
        self,
        dataloader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        grad_clip: float = 1.0,
        use_amp: bool = False,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, float], Optional[torch.Tensor]]:
        """
        Train one epoch with online topology adaptation.

        Returns:
            losses: Dict of loss averages
            hidden_state: Updated hidden state (for temporal continuity)
        """
        self.model.train()
        self.topology_detector.train()

        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        meters = {}
        for key in [
            "total",
            "state",
            "parameter",
            "physics",
            "topo_aux",
            "adaptation_ms",
        ]:
            meters[key] = _AverageMeter()

        pbar = tqdm(dataloader, desc="Topo-Aware Epoch %d" % epoch)

        for batch in pbar:
            edge_index = batch["topology"]["edge_index"].to(self.device)
            edge_attr = batch["topology"]["edge_attr"].to(self.device)
            true_states = {
                k: v.to(self.device) for k, v in batch["true_states"].items()
            }
            true_params = {k: v.to(self.device) for k, v in batch["parameters"].items()}

            measurements, obs_mask = self._merge_measurements(batch)
            measurements = {k: v.to(self.device) for k, v in measurements.items()}
            obs_mask = obs_mask.to(self.device)

            pmu_mask_batch = batch["pmu_mask"].to(self.device)
            scada_mask_batch = batch["scada_mask"].to(self.device)
            has_topo = batch.get("has_topology_change")
            if has_topo is not None:
                has_topo = has_topo.to(self.device)

            has_topo_scalar = has_topo.any().item() if has_topo is not None else False

            if has_topo_scalar:
                affected_nodes, adapt_ms = self._handle_topology_change(
                    measurements, edge_index, True, hidden_state
                )
            else:
                affected_nodes, adapt_ms = self._handle_topology_change(
                    measurements, edge_index, False, hidden_state
                )

            optimizer.zero_grad()

            with torch.cuda.amp.autocast() if use_amp else _nullcontext():
                pred_states, pred_params = self.model(
                    measurements, edge_index, edge_attr=edge_attr, obs_mask=obs_mask
                )

                if (
                    isinstance(true_params["r_line"], torch.Tensor)
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
                if self.topology_detector is not None:
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
            meters["adaptation_ms"].update(adapt_ms, 1)

            pbar.set_postfix(
                {
                    "loss": "%.4f" % meters["total"].avg,
                    "adapt_ms": "%.1f" % meters["adaptation_ms"].avg,
                    "topo_det": self.topo_stats["num_detections"],
                }
            )

        return {k: v.avg for k, v in meters.items()}, hidden_state

    def train_topology_curriculum(
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
        topo_start_epoch: int = 30,
    ) -> Dict:
        """
        Train with topology curriculum: no topology changes for first epochs,
        then gradually introduce them.
        """
        from pathlib import Path
        import time as time_mod

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        stage_epochs = {
            1: staged_config.get("stage1_epochs", 30),
            2: staged_config.get("stage2_epochs", 30),
            3: staged_config.get("stage3_epochs", 140),
        }
        total_epochs = sum(stage_epochs.values())
        global_epoch = 0
        es_counter = 0
        start_time = time_mod.time()

        for stage in [1, 2, 3]:
            num_epochs = stage_epochs[stage]
            print("\n" + "=" * 60)
            print("TOPO CURRICULUM STAGE %d: %d epochs" % (stage, num_epochs))
            print("=" * 60)

            status = self._set_stage(stage)
            print("  Frozen: %s" % (status["frozen"] if status["frozen"] else "None"))
            n_trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
            print("  Trainable params: %d" % n_trainable)

            optimizer = self._create_optimizer(stage)
            scheduler = self._create_scheduler(optimizer, num_epochs)

            for local_epoch in range(1, num_epochs + 1):
                global_epoch += 1

                use_topology = global_epoch >= topo_start_epoch

                if use_topology:
                    train_losses, _ = self.train_epoch_with_topology(
                        train_loader,
                        criterion,
                        optimizer,
                        global_epoch,
                        grad_clip=grad_clip,
                        use_amp=use_amp,
                    )
                else:
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

                topo_metrics = self.get_topology_metrics()

                print(
                    "\n  Epoch %d/%d (Stage %d, Topo=%s)  LR: %.2e"
                    % (global_epoch, total_epochs, stage, use_topology, current_lr)
                )
                print("    Train Loss: %.6f" % train_losses["total"])
                print("    Val Loss:   %.6f" % val_results["total"])
                if "v_mag_rmse" in val_results:
                    print("    Val V_mag RMSE: %.6f" % val_results["v_mag_rmse"])
                if use_topology:
                    print(
                        "    Topo Precision: %.3f  Recall: %.3f  F1: %.3f"
                        % (
                            topo_metrics["precision"],
                            topo_metrics["recall"],
                            topo_metrics["f1"],
                        )
                    )
                    if topo_metrics["avg_adaptation_ms"] > 0:
                        print(
                            "    Avg Adaptation: %.1f ms"
                            % topo_metrics["avg_adaptation_ms"]
                        )

                if writer is not None:
                    writer.add_scalar("Loss/train", train_losses["total"], global_epoch)
                    writer.add_scalar("Loss/val", val_results["total"], global_epoch)
                    writer.add_scalar("Stage", stage, global_epoch)
                    writer.add_scalar("LR", current_lr, global_epoch)
                    writer.add_scalar(
                        "Topology/use_topology", float(use_topology), global_epoch
                    )
                    if use_topology:
                        for k, v in topo_metrics.items():
                            if isinstance(v, float):
                                writer.add_scalar("Topology/%s" % k, v, global_epoch)

                if swanlab_run is not None:
                    log_dict = {
                        "train/loss_total": train_losses["total"],
                        "val/loss_total": val_results["total"],
                        "train/stage": stage,
                        "train/global_epoch": global_epoch,
                        "topology/use_topology": use_topology,
                    }
                    if use_topology:
                        log_dict["topology/precision"] = topo_metrics["precision"]
                        log_dict["topology/recall"] = topo_metrics["recall"]
                        log_dict["topology/f1"] = topo_metrics["f1"]
                    swanlab_run.log(log_dict, step=global_epoch)

                if val_results["total"] < self.best_val_loss - early_stopping_min_delta:
                    self.best_val_loss = val_results["total"]
                    self.best_epoch = global_epoch
                    es_counter = 0
                    self._save_checkpoint(
                        optimizer,
                        global_epoch,
                        self.best_val_loss,
                        save_path / "best_model.pt",
                    )
                else:
                    es_counter += 1

                if global_epoch % save_freq == 0:
                    self._save_checkpoint(
                        optimizer,
                        global_epoch,
                        val_results["total"],
                        save_path / ("checkpoint_epoch_%d.pt" % global_epoch),
                    )

                if es_counter >= early_stopping_patience:
                    print("\n  Early stopping at epoch %d" % global_epoch)
                    break

        elapsed = time_mod.time() - start_time
        print("\n" + "=" * 60)
        print("Topology curriculum training complete in %.2f hours" % (elapsed / 3600))
        print("Best val loss: %.6f at epoch %d" % (self.best_val_loss, self.best_epoch))
        print("=" * 60)

        final_topo = self.get_topology_metrics()

        summary = {
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "total_epochs": global_epoch,
            "training_time_hours": elapsed / 3600,
            "topology_precision": final_topo["precision"],
            "topology_recall": final_topo["recall"],
            "topology_f1": final_topo["f1"],
            "avg_adaptation_ms": final_topo["avg_adaptation_ms"],
        }

        if swanlab_run is not None:
            swanlab_run.log({"summary/" + k: v for k, v in summary.items()})
            swanlab_run.finish()

        return summary

    def get_topology_metrics(self) -> Dict[str, float]:
        """Compute topology detection metrics from accumulated stats."""
        s = self.topo_stats
        tp = s["num_true_positives"]
        fp = s["num_false_positives"]
        fn = s["num_false_negatives"]

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)

        avg_adapt_ms = 0.0
        if len(s["adaptation_times_ms"]) > 0:
            avg_adapt_ms = sum(s["adaptation_times_ms"]) / len(s["adaptation_times_ms"])

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_detections": s["num_detections"],
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "avg_adaptation_ms": avg_adapt_ms,
        }
