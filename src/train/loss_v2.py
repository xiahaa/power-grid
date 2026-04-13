"""
Multi-Rate Estimation Loss for DSSE

Extends JointEstimationLoss with:
1. Separate PMU/SCADA stream losses with different weights
2. Hierarchical physics constraint integration
3. Topology change detection auxiliary loss
4. Cross-attention alignment quality loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiRateEstimationLoss(nn.Module):
    """
    Loss for multi-rate Mamba fusion DSSE.

    L = w_state * L_state
      + w_param * L_param
      + w_physics * L_physics_hier
      + w_smooth * L_smooth
      + w_pmu * L_pmu_align
      + w_scada * L_scada_align
      + w_topo * L_topo_aux
    """

    def __init__(
        self,
        state_weight: float = 1.0,
        parameter_weight: float = 0.5,
        physics_weight: float = 0.0001,
        temporal_smoothness: float = 0.01,
        pmu_loss_weight: float = 1.0,
        scada_loss_weight: float = 0.5,
        topo_aux_weight: float = 0.1,
        feeder_map: Optional[Dict[int, list]] = None,
        feeder_loss_weights: Optional[Dict[str, object]] = None,
    ):
        super().__init__()
        self.state_weight = state_weight
        self.parameter_weight = parameter_weight
        self.physics_weight = physics_weight
        self.temporal_smoothness = temporal_smoothness
        self.pmu_loss_weight = pmu_loss_weight
        self.scada_loss_weight = scada_loss_weight
        self.topo_aux_weight = topo_aux_weight
        self.feeder_map = feeder_map or {}
        self.feeder_loss_weights = feeder_loss_weights or {}

    @property
    def use_parameter_loss(self) -> bool:
        return self.parameter_weight > 0.0

    def _node_weight_tensor(
        self,
        key: str,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.feeder_map:
            return None

        enabled = self.feeder_loss_weights.get("enabled", False)
        if not enabled:
            return None

        key_weights = self.feeder_loss_weights.get(key)
        if not isinstance(key_weights, dict):
            return None

        default_weight = float(self.feeder_loss_weights.get("default_weight", 1.0))
        weights = torch.full((num_nodes,), default_weight, device=device, dtype=dtype)
        one_indexed = bool(self.feeder_loss_weights.get("one_indexed_bus_numbers", True))

        for feeder_id, buses in self.feeder_map.items():
            feeder_weight = key_weights.get(str(feeder_id), key_weights.get(feeder_id))
            if feeder_weight is None:
                continue
            for bus in buses:
                idx = bus - 1 if one_indexed else bus
                if 0 <= idx < num_nodes:
                    weights[idx] = float(feeder_weight)

        return weights

    def state_loss(
        self,
        pred_states: Dict[str, torch.Tensor],
        true_states: Dict[str, torch.Tensor],
        pmu_mask: Optional[torch.Tensor] = None,
        scada_mask: Optional[torch.Tensor] = None,
        true_state_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        device = pred_states["v_mag"].device
        loss = torch.tensor(0.0, device=device)

        for key in ["v_mag", "v_ang"]:
            pred = pred_states[key]
            target = true_states[key]

            base_loss = F.mse_loss(pred, target, reduction="none")
            effective_weight = torch.ones_like(base_loss)

            valid_mask = None
            if true_state_masks is not None and key in true_state_masks:
                valid_mask = true_state_masks[key].to(pred.device)
                if valid_mask.dim() < pred.dim():
                    while valid_mask.dim() < pred.dim():
                        valid_mask = valid_mask.unsqueeze(0)
                if valid_mask.shape != pred.shape:
                    valid_mask = valid_mask.expand_as(pred)
                effective_weight = effective_weight * valid_mask.float()

            node_weights = self._node_weight_tensor(
                key, pred.shape[-1], pred.device, base_loss.dtype
            )
            if node_weights is not None:
                if pred.dim() == 2:
                    effective_weight = effective_weight * node_weights.unsqueeze(0)
                elif pred.dim() == 3:
                    effective_weight = effective_weight * node_weights.view(1, 1, -1)

            if pmu_mask is not None and key == "v_mag":
                pmu_mask_reduced = pmu_mask
                if pmu_mask.dim() > pred.dim():
                    pmu_mask_reduced = pmu_mask[:, -1, :]
                elif pmu_mask.dim() < pred.dim():
                    pmu_mask_reduced = pmu_mask.unsqueeze(-1).expand_as(pred)

                if pmu_mask_reduced.shape != pred.shape:
                    aligned_mask = torch.zeros_like(pred, dtype=torch.bool)
                    min_last_dim = min(pmu_mask_reduced.shape[-1], pred.shape[-1])

                    if pred.dim() == 2:
                        aligned_mask[:, :min_last_dim] = pmu_mask_reduced[
                            :, :min_last_dim
                        ]
                    elif pred.dim() == 3:
                        aligned_mask[:, :, :min_last_dim] = pmu_mask_reduced[
                            ..., :min_last_dim
                        ]
                    else:
                        raise ValueError(
                            "Unsupported prediction tensor rank for PMU mask alignment"
                        )
                    pmu_mask_reduced = aligned_mask

                pmu_weight = pmu_mask_reduced.float() * self.pmu_loss_weight
                scada_weight = (1.0 - pmu_mask_reduced.float()) * self.scada_loss_weight
                combined_weight = pmu_weight + scada_weight
                effective_weight = effective_weight * combined_weight

            denom = effective_weight.sum().clamp_min(1.0)
            loss = loss + (base_loss * effective_weight).sum() / denom

        return loss

    def parameter_loss(
        self,
        pred_params: Optional[Dict[str, torch.Tensor]],
        true_params: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if (
            not self.use_parameter_loss
            or pred_params is None
            or true_params is None
            or "r_line" not in pred_params
            or "r_line" not in true_params
        ):
            device = None
            if pred_params is not None and "r_line" in pred_params:
                device = pred_params["r_line"].device
            elif true_params is not None and "r_line" in true_params:
                device = true_params["r_line"].device
            return torch.tensor(0.0, device=device)

        device = pred_params["r_line"].device
        loss = torch.tensor(0.0, device=device)

        for key in ["r_line", "x_line"]:
            pred = pred_params[key]
            target = true_params[key]

            if pred.shape != target.shape:
                if pred.shape[0] == target.shape[0]:
                    min_len = min(pred.shape[-1], target.shape[-1])
                    pred = pred[..., :min_len]
                    target = target[..., :min_len]
                else:
                    continue

            pred_log = torch.log(pred.clamp_min(1e-8))
            target_log = torch.log(target.clamp_min(1e-8))
            loss += F.mse_loss(pred_log, target_log)

        return loss

    def temporal_smoothness_loss(
        self,
        states_sequence: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if states_sequence is None or "v_mag" not in states_sequence:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=states_sequence["v_mag"].device)
        for key, values in states_sequence.items():
            if values.dim() >= 2:
                diff = values[:, 1:] - values[:, :-1]
                loss = loss + (diff**2).mean()

        return loss

    def topology_aux_loss(
        self,
        anomaly_scores: torch.Tensor,
        has_topology_change: torch.Tensor,
    ) -> torch.Tensor:
        if anomaly_scores is None or has_topology_change is None:
            device = None
            if anomaly_scores is not None:
                device = anomaly_scores.device
            elif has_topology_change is not None:
                device = has_topology_change.device
            return torch.tensor(0.0, device=device)

        targets = has_topology_change.float().to(anomaly_scores.device)
        if targets.dim() < anomaly_scores.dim():
            targets = targets.unsqueeze(-1).expand_as(anomaly_scores)

        scores = anomaly_scores.clamp(min=1e-6, max=1.0 - 1e-6)
        return F.binary_cross_entropy(scores, targets)

    def forward(
        self,
        pred_states: Dict[str, torch.Tensor],
        true_states: Dict[str, torch.Tensor],
        pred_params: Optional[Dict[str, torch.Tensor]],
        true_params: Optional[Dict[str, torch.Tensor]],
        hierarchical_physics_loss: torch.Tensor,
        anomaly_scores: Optional[torch.Tensor] = None,
        has_topology_change: Optional[torch.Tensor] = None,
        pmu_mask: Optional[torch.Tensor] = None,
        scada_mask: Optional[torch.Tensor] = None,
        true_state_masks: Optional[Dict[str, torch.Tensor]] = None,
        states_sequence: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple:
        device = pred_states["v_mag"].device
        l_state = self.state_loss(
            pred_states,
            true_states,
            pmu_mask,
            scada_mask,
            true_state_masks=true_state_masks,
        )
        l_param = self.parameter_loss(pred_params, true_params)
        if isinstance(hierarchical_physics_loss, torch.Tensor):
            l_physics = hierarchical_physics_loss.to(device)
        else:
            l_physics = torch.tensor(float(hierarchical_physics_loss), device=device)
        l_smooth = self.temporal_smoothness_loss(states_sequence)
        l_smooth = l_smooth.to(device)
        l_topo = self.topology_aux_loss(anomaly_scores, has_topology_change)
        l_topo = l_topo.to(device)

        total_loss = (
            self.state_weight * l_state
            + self.parameter_weight * l_param
            + self.physics_weight * l_physics
            + self.temporal_smoothness * l_smooth
            + self.topo_aux_weight * l_topo
        )

        loss_dict = {
            "total": total_loss.item(),
            "state": l_state.item(),
            "parameter": l_param.item(),
            "physics": l_physics.item()
            if isinstance(l_physics, torch.Tensor)
            else l_physics,
            "smoothness": l_smooth.item()
            if isinstance(l_smooth, torch.Tensor)
            else l_smooth,
            "topo_aux": l_topo.item() if isinstance(l_topo, torch.Tensor) else l_topo,
        }

        return total_loss, loss_dict
