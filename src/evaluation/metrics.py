"""
Comprehensive Evaluation Metrics for DSSE

Metrics categories:
  1. State estimation: RMSE, MAE, MAPE for v_mag, v_ang
  2. Parameter estimation: RMSE, MAE, MAPE for r_line, x_line
  3. Topology detection: Precision, Recall, F1, confusion matrix
  4. Adaptation speed: Detection latency, reset latency, convergence time
  5. Physics violation: Voltage limit violations, power balance error
  6. Robustness: Performance under missing data, bad data, rate variation
  7. Per-bus / per-feeder breakdown
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class StateEstimationMetrics:
    """Voltage magnitude and angle estimation metrics."""

    @staticmethod
    def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
        valid = ~(torch.isnan(pred) | torch.isnan(target))
        if valid.sum() == 0:
            return float("nan")
        return torch.sqrt(((pred[valid] - target[valid]) ** 2).mean()).item()

    @staticmethod
    def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
        valid = ~(torch.isnan(pred) | torch.isnan(target))
        if valid.sum() == 0:
            return float("nan")
        return torch.abs(pred[valid] - target[valid]).mean().item()

    @staticmethod
    def mape(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> float:
        valid = ~(torch.isnan(pred) | torch.isnan(target))
        if valid.sum() == 0:
            return float("nan")
        return (
            (torch.abs((pred[valid] - target[valid]) / (target[valid] + epsilon)) * 100)
            .mean()
            .item()
        )

    @staticmethod
    def max_error(pred: torch.Tensor, target: torch.Tensor) -> float:
        valid = ~(torch.isnan(pred) | torch.isnan(target))
        if valid.sum() == 0:
            return float("nan")
        return torch.abs(pred[valid] - target[valid]).max().item()

    @staticmethod
    def percentile_error(
        pred: torch.Tensor, target: torch.Tensor, p: float = 95.0
    ) -> float:
        valid = ~(torch.isnan(pred) | torch.isnan(target))
        if valid.sum() == 0:
            return float("nan")
        errors = torch.abs(pred[valid] - target[valid]).flatten()
        return torch.quantile(errors, p / 100.0).item()

    @staticmethod
    def compute_all(
        pred_states: Dict[str, torch.Tensor],
        true_states: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        metrics = {}
        for key in ["v_mag", "v_ang"]:
            if key in pred_states and key in true_states:
                pred, target = pred_states[key], true_states[key]
                metrics["%s_rmse" % key] = StateEstimationMetrics.rmse(pred, target)
                metrics["%s_mae" % key] = StateEstimationMetrics.mae(pred, target)
                metrics["%s_mape" % key] = StateEstimationMetrics.mape(pred, target)
                metrics["%s_max_error" % key] = StateEstimationMetrics.max_error(
                    pred, target
                )
                metrics["%s_p95_error" % key] = StateEstimationMetrics.percentile_error(
                    pred, target
                )
        return metrics


class ParameterEstimationMetrics:
    """Line parameter estimation metrics."""

    @staticmethod
    def compute_all(
        pred_params: Dict[str, torch.Tensor],
        true_params: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        metrics = {}
        if pred_params is None or true_params is None:
            return metrics
        for key in ["r_line", "x_line"]:
            if key in pred_params and key in true_params:
                pred, target = pred_params[key], true_params[key]
                metrics["%s_rmse" % key] = torch.sqrt(
                    ((pred - target) ** 2).mean()
                ).item()
                metrics["%s_mae" % key] = torch.abs(pred - target).mean().item()
                metrics["%s_mape" % key] = (
                    (torch.abs((pred - target) / (target + 1e-8)) * 100).mean().item()
                )
        return metrics


class TopologyDetectionMetrics:
    """Topology change detection quality metrics."""

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.detection_latencies_ms = []
        self.false_alarm_latencies_ms = []

    def update(
        self,
        predicted_change: bool,
        actual_change: bool,
        latency_ms: float = 0.0,
    ):
        if predicted_change and actual_change:
            self.tp += 1
            self.detection_latencies_ms.append(latency_ms)
        elif predicted_change and not actual_change:
            self.fp += 1
            self.false_alarm_latencies_ms.append(latency_ms)
        elif not predicted_change and actual_change:
            self.fn += 1
        else:
            self.tn += 1

    def precision(self) -> float:
        return self.tp / max(1, self.tp + self.fp)

    def recall(self) -> float:
        return self.tp / max(1, self.tp + self.fn)

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / max(1e-8, p + r)

    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / max(1, total)

    def avg_detection_latency_ms(self) -> float:
        if len(self.detection_latencies_ms) == 0:
            return 0.0
        return sum(self.detection_latencies_ms) / len(self.detection_latencies_ms)

    def compute_summary(self) -> Dict[str, float]:
        return {
            "topo_precision": self.precision(),
            "topo_recall": self.recall(),
            "topo_f1": self.f1(),
            "topo_accuracy": self.accuracy(),
            "topo_tp": float(self.tp),
            "topo_fp": float(self.fp),
            "topo_fn": float(self.fn),
            "topo_tn": float(self.tn),
            "topo_avg_detection_latency_ms": self.avg_detection_latency_ms(),
        }

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.detection_latencies_ms = []
        self.false_alarm_latencies_ms = []


class PhysicsViolationMetrics:
    """Physics constraint violation metrics."""

    def __init__(self, v_min: float = 0.95, v_max: float = 1.05):
        self.v_min = v_min
        self.v_max = v_max

    def voltage_violation_rate(self, v_mag: torch.Tensor) -> Dict[str, float]:
        valid = torch.isfinite(v_mag)
        if valid.sum() == 0:
            return {
                "voltage_violation_rate": float("nan"),
                "voltage_under_voltage_rate": float("nan"),
                "voltage_over_voltage_rate": float("nan"),
                "voltage_max_deviation": float("nan"),
            }

        v_valid = v_mag[valid]
        violations = (v_valid < self.v_min) | (v_valid > self.v_max)
        num_violations = violations.sum().item()
        total = v_valid.numel()
        rate = num_violations / max(1, total)
        under = (v_valid < self.v_min).sum().item()
        over = (v_valid > self.v_max).sum().item()

        return {
            "voltage_violation_rate": rate,
            "voltage_under_voltage_rate": under / max(1, total),
            "voltage_over_voltage_rate": over / max(1, total),
            "voltage_max_deviation": max(
                abs(v_valid.min().item() - self.v_min),
                abs(v_valid.max().item() - self.v_max),
            ),
        }

    def power_balance_error(
        self,
        p_bus: torch.Tensor,
        q_bus: torch.Tensor,
    ) -> Dict[str, float]:
        p_imbalance = p_bus.sum(dim=-1)
        q_imbalance = q_bus.sum(dim=-1)

        return {
            "power_balance_p_mean": p_imbalance.abs().mean().item(),
            "power_balance_q_mean": q_imbalance.abs().mean().item(),
            "power_balance_p_max": p_imbalance.abs().max().item(),
            "power_balance_q_max": q_imbalance.abs().max().item(),
        }

    def compute_all(
        self,
        states: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        metrics = {}

        if "v_mag" in states:
            metrics.update(self.voltage_violation_rate(states["v_mag"]))

        if "p_bus" in states and "q_bus" in states:
            metrics.update(self.power_balance_error(states["p_bus"], states["q_bus"]))

        return metrics


class PerBusMetrics:
    """Per-bus estimation accuracy breakdown."""

    @staticmethod
    def compute_per_bus(
        pred_states: Dict[str, torch.Tensor],
        true_states: Dict[str, torch.Tensor],
    ) -> Dict[str, List[float]]:
        result = {}
        for key in ["v_mag", "v_ang"]:
            if key in pred_states and key in true_states:
                pred, target = pred_states[key], true_states[key]
                errors = torch.abs(pred - target)
                valid = torch.isfinite(pred) & torch.isfinite(target)
                masked_errors = torch.where(errors.isfinite() & valid, errors, torch.nan)
                result["%s_per_bus_mae" % key] = torch.nanmean(
                    masked_errors, dim=0
                ).tolist()
                masked_max = torch.where(
                    valid, errors, torch.full_like(errors, float("-inf"))
                )
                max_vals = masked_max.max(dim=0).values
                max_vals[max_vals == float("-inf")] = float("nan")
                result["%s_per_bus_max" % key] = max_vals.tolist()
        return result

    @staticmethod
    def compute_worst_buses(
        pred_states: Dict[str, torch.Tensor],
        true_states: Dict[str, torch.Tensor],
        n_worst: int = 10,
    ) -> Dict[str, List[Tuple[int, float]]]:
        worst = {}
        for key in ["v_mag", "v_ang"]:
            if key in pred_states and key in true_states:
                pred, target = pred_states[key], true_states[key]
                errors = torch.abs(pred - target)
                valid = torch.isfinite(pred) & torch.isfinite(target)
                masked_errors = torch.where(valid, errors, torch.nan)
                mae_per_bus = torch.nanmean(masked_errors, dim=0)
                score = torch.nan_to_num(mae_per_bus, nan=float("-inf"))
                topk = torch.topk(score, min(n_worst, score.shape[-1]))
                worst["%s_worst_%d_buses" % (key, n_worst)] = list(
                    zip(topk.indices.tolist(), mae_per_bus[topk.indices].tolist())
                )
        return worst


class PerFeederMetrics:
    """Per-feeder estimation accuracy breakdown."""

    def __init__(self, feeder_map: Dict[int, List[int]] = None):
        self.feeder_map = feeder_map or {}

    def compute_per_feeder(
        self,
        pred_states: Dict[str, torch.Tensor],
        true_states: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[int, float]]:
        result = {}
        for key in ["v_mag", "v_ang"]:
            if key not in pred_states or key not in true_states:
                continue
            pred, target = pred_states[key], true_states[key]
            feeder_metrics = {}
            for feeder_id, buses in self.feeder_map.items():
                if len(buses) == 0:
                    continue
                bus_idx = torch.tensor(buses, dtype=torch.long)
                bus_idx = bus_idx[bus_idx < pred.shape[-1]]
                if len(bus_idx) == 0:
                    continue
                f_pred = pred[..., bus_idx]
                f_target = target[..., bus_idx]
                valid = torch.isfinite(f_pred) & torch.isfinite(f_target)
                if valid.sum() == 0:
                    feeder_metrics[feeder_id] = float("nan")
                else:
                    feeder_metrics[feeder_id] = torch.sqrt(
                        ((f_pred[valid] - f_target[valid]) ** 2).mean()
                    ).item()
            result["%s_feeder_rmse" % key] = feeder_metrics
        return result


class RobustnessMetrics:
    """Metrics for robustness tests (missing data, bad data, rate variation)."""

    @staticmethod
    def missing_data_sensitivity(
        pred_states: Dict[str, torch.Tensor],
        true_states: Dict[str, torch.Tensor],
        obs_masks: Dict[str, torch.Tensor],
        thresholds: List[float] = None,
    ) -> Dict[str, float]:
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.5, 0.7]

        result = {}
        for key in ["v_mag"]:
            if key not in pred_states or key not in true_states:
                continue
            errors = torch.abs(pred_states[key] - true_states[key])

            if "scada" in obs_masks and obs_masks["scada"] is not None:
                mask = obs_masks["scada"]
                if mask.dim() < errors.dim():
                    mask = mask.unsqueeze(1)
                    mask = mask.expand_as(errors)
                obs_rate = mask.float().mean().item()

                for thresh in thresholds:
                    if obs_rate >= thresh:
                        result["%s_rmse_at_obs_%.1f" % (key, thresh)] = torch.sqrt(
                            (errors**2).mean()
                        ).item()
        return result

    @staticmethod
    def compute_all(
        pred_states: Dict[str, torch.Tensor],
        true_states: Dict[str, torch.Tensor],
        obs_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        metrics = {}
        for key in ["v_mag", "v_ang"]:
            if key in pred_states and key in true_states:
                pred, target = pred_states[key], true_states[key]
                valid = torch.isfinite(pred) & torch.isfinite(target)
                if valid.sum() == 0:
                    metrics["%s_rmse" % key] = float("nan")
                    metrics["%s_mae" % key] = float("nan")
                else:
                    diff = pred[valid] - target[valid]
                    metrics["%s_rmse" % key] = torch.sqrt((diff**2).mean()).item()
                    metrics["%s_mae" % key] = diff.abs().mean().item()

        if obs_masks is not None:
            metrics.update(
                RobustnessMetrics.missing_data_sensitivity(
                    pred_states, true_states, obs_masks
                )
            )
        return metrics


class DSSEEvaluator:
    """
    Comprehensive DSSE evaluator combining all metric classes.
    """

    def __init__(
        self,
        feeder_map: Dict[int, List[int]] = None,
        v_min: float = 0.95,
        v_max: float = 1.05,
        include_parameters: bool = True,
    ):
        self.state_metrics = StateEstimationMetrics()
        self.param_metrics = ParameterEstimationMetrics() if include_parameters else None
        self.topo_metrics = TopologyDetectionMetrics()
        self.physics_metrics = PhysicsViolationMetrics(v_min, v_max)
        self.per_bus_metrics = PerBusMetrics()
        self.per_feeder_metrics = PerFeederMetrics(feeder_map)
        self.include_parameters = include_parameters

    def evaluate(
        self,
        pred_states: Dict[str, torch.Tensor],
        true_states: Dict[str, torch.Tensor],
        pred_params: Optional[Dict[str, torch.Tensor]] = None,
        true_params: Optional[Dict[str, torch.Tensor]] = None,
        predicted_topo_change: Optional[bool] = None,
        actual_topo_change: Optional[bool] = None,
        detection_latency_ms: float = 0.0,
    ) -> Dict[str, float]:
        results = {}
        results.update(self.state_metrics.compute_all(pred_states, true_states))
        if self.include_parameters and self.param_metrics is not None:
            results.update(self.param_metrics.compute_all(pred_params, true_params))
        results.update(self.physics_metrics.compute_all(pred_states))

        if predicted_topo_change is not None and actual_topo_change is not None:
            self.topo_metrics.update(
                predicted_topo_change, actual_topo_change, detection_latency_ms
            )
            results.update(self.topo_metrics.compute_summary())

        return results

    def evaluate_batch(
        self,
        all_pred_states: List[Dict[str, torch.Tensor]],
        all_true_states: List[Dict[str, torch.Tensor]],
        all_pred_params: Optional[List[Dict[str, torch.Tensor]]] = None,
        all_true_params: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, float]:
        if not all_pred_states or not all_true_states:
            return {"error": "no_valid_batches"}

        pred_cat = {
            k: torch.cat([d[k] for d in all_pred_states]) for k in all_pred_states[0]
        }
        true_cat = {
            k: torch.cat([d[k] for d in all_true_states]) for k in all_true_states[0]
        }
        pred_p = None
        true_p = None
        if self.include_parameters and all_pred_params and all_true_params:
            try:
                pred_p = {
                    k: torch.cat([d[k] for d in all_pred_params])
                    for k in all_pred_params[0]
                }
                true_p = {
                    k: torch.cat([d[k] for d in all_true_params])
                    for k in all_true_params[0]
                }
            except (RuntimeError, IndexError, TypeError):
                pred_p = None
                true_p = None

        if pred_p is not None and true_p is not None:
            for key in ["r_line", "x_line"]:
                if key in pred_p and key in true_p:
                    if pred_p[key].shape[-1] != true_p[key].shape[-1]:
                        min_len = min(pred_p[key].shape[-1], true_p[key].shape[-1])
                        pred_p[key] = pred_p[key][..., :min_len]
                        true_p[key] = true_p[key][..., :min_len]

        if pred_p is not None and true_p is not None:
            results = self.evaluate(pred_cat, true_cat, pred_p, true_p)
        else:
            results = self.evaluate(pred_cat, true_cat, None, None)
        results.update(self.per_bus_metrics.compute_worst_buses(pred_cat, true_cat))
        results.update(self.per_feeder_metrics.compute_per_feeder(pred_cat, true_cat))

        return results

    def get_topology_summary(self) -> Dict[str, float]:
        return self.topo_metrics.compute_summary()

    def reset_topology(self):
        self.topo_metrics.reset()

    def format_results(self, results: Dict[str, float]) -> str:
        lines = ["=" * 60, "DSSE Evaluation Results", "=" * 60]

        state_keys = [
            k for k in results if k.startswith("v_mag") or k.startswith("v_ang")
        ]
        if state_keys:
            lines.append("\nState Estimation:")
            for k in sorted(state_keys):
                if isinstance(results[k], float):
                    lines.append("  %s: %.6f" % (k, results[k]))

        param_keys = []
        if self.include_parameters:
            param_keys = [
                k for k in results if k.startswith("r_line") or k.startswith("x_line")
            ]
        if param_keys:
            lines.append("\nParameter Estimation:")
            for k in sorted(param_keys):
                if isinstance(results[k], float):
                    lines.append("  %s: %.6f" % (k, results[k]))

        physics_keys = [k for k in results if "voltage" in k or "power_balance" in k]
        if physics_keys:
            lines.append("\nPhysics Violations:")
            for k in sorted(physics_keys):
                if isinstance(results[k], float):
                    lines.append("  %s: %.6f" % (k, results[k]))

        topo_keys = [k for k in results if "topo" in k]
        if topo_keys:
            lines.append("\nTopology Detection:")
            for k in sorted(topo_keys):
                if isinstance(results[k], float):
                    lines.append("  %s: %.4f" % (k, results[k]))

        feeder_keys = [k for k in results if "feeder" in k]
        if feeder_keys:
            lines.append("\nPer-Feeder RMSE:")
            for k in sorted(feeder_keys):
                val = results[k]
                if isinstance(val, dict):
                    lines.append("  %s:" % k)
                    for fid, rmse in sorted(val.items()):
                        lines.append("    Feeder %d: %.6f" % (fid, rmse))

        lines.append("=" * 60)
        return "\n".join(lines)
