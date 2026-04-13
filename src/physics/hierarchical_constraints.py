"""
Hierarchical Physics-Informed Constraints for Multi-Scale DSSE

Extends single-level bus physics constraints to three hierarchical levels:
1. Bus level: Individual KCL/KVL (123 buses)
2. Feeder level: Power balance per feeder (6 feeders)
3. Substation level: Overall system balance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .constraints import PhysicsInformedLayer


def prepare_states_for_physics(
    states: Dict[str, torch.Tensor],
    measurements: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Augment state tensors with latest bus injections when available."""
    physics_states = dict(states)

    if measurements is None:
        return physics_states

    for key in ["p_bus", "q_bus"]:
        if key in physics_states or key not in measurements:
            continue

        values = measurements[key]
        if values is None:
            continue

        if values.dim() > states["v_mag"].dim():
            physics_states[key] = values[:, -1, :]
        else:
            physics_states[key] = values

    return physics_states


class NoOpPhysicsConstraints(nn.Module):
    """Physics constraint stub used for ablation runs."""

    def forward(
        self,
        states: Dict[str, torch.Tensor],
        parameters: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        measurements: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        device = states["v_mag"].device
        zero = torch.tensor(0.0, device=device)
        return {
            "bus": zero,
            "feeder": zero,
            "substation": zero,
            "total": zero,
        }


class FeederAssignment:
    """Manages feeder-to-bus mapping for hierarchical constraints."""

    def __init__(self, feeder_map: Dict[int, List[int]] = None):
        self.feeder_map = feeder_map or {}

    def get_feeder_buses(self, feeder_id: int) -> List[int]:
        return self.feeder_map.get(feeder_id, [])

    def get_num_feeders(self) -> int:
        return len(self.feeder_map)

    def get_all_buses(self) -> List[int]:
        return [b for buses in self.feeder_map.values() for b in buses]


class HierarchicalPhysicsConstraints(nn.Module):
    """
    Multi-scale physics constraints for three hierarchical levels.

    Architecture:
        Bus Level (KCL/KVL per bus)
            --> Standard power flow mismatch
        Feeder Level (Power balance per feeder)
            --> Sum of bus injections = feeder head
        Substation Level (Overall balance)
            --> Sum of all feeders = substation supply
    """

    def __init__(
        self,
        feeder_map: Optional[Dict[int, List[int]]] = None,
        num_feeders: int = 6,
        bus_weight: float = 1.0,
        feeder_weight: float = 0.5,
        substation_weight: float = 0.2,
        voltage_limits: Tuple[float, float] = (0.95, 1.05),
        constraint_type: str = "soft",
        projection_method: str = "gradient_descent",
        max_iterations: int = 20,
        tolerance: float = 1e-4,
        power_balance_weight: float = 10.0,
        slack_bus: Optional[int] = None,
        hierarchical_enabled: bool = True,
    ):
        """
        Args:
            feeder_map: Mapping feeder_id -> list of bus indices.
            num_feeders: Number of feeders (default 6)
            bus_weight: Weight for bus-level constraints
            feeder_weight: Weight for feeder-level constraints
            substation_weight: Weight for substation-level constraints
            voltage_limits: (v_min, v_max) voltage limits in p.u.
        """
        super().__init__()

        if feeder_map is None:
            feeder_map = self._build_default_feeder_map(num_feeders)

        self.feeder_assignment = FeederAssignment(feeder_map)
        self.num_feeders = self.feeder_assignment.get_num_feeders()
        self.bus_weight = bus_weight
        self.feeder_weight = feeder_weight
        self.substation_weight = substation_weight
        self.hierarchical_enabled = hierarchical_enabled
        self.v_min, self.v_max = voltage_limits
        self.bus_physics = PhysicsInformedLayer(
            constraint_type=constraint_type,
            projection_method=projection_method,
            max_iterations=max_iterations,
            tolerance=tolerance,
            voltage_limits=voltage_limits,
            power_balance_weight=power_balance_weight,
            slack_bus=slack_bus,
        )

    def _build_default_feeder_map(self, num_feeders: int) -> Dict[int, List[int]]:
        """Build default feeder grouping for IEEE 123-bus system."""
        buses_per_feeder = 123 // num_feeders
        feeder_map = {}
        for f in range(num_feeders):
            start = f * buses_per_feeder
            end = start + buses_per_feeder
            feeder_map[f] = list(range(start, end))
        return feeder_map

    def forward(
        self,
        states: Dict[str, torch.Tensor],
        parameters: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        measurements: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical physics constraints at all three levels.

        Args:
            states: Dict with 'v_mag', 'v_ang', and optionally 'p_bus', 'q_bus'
            parameters: Dict with 'r_line', 'x_line'
            edge_index: [2, num_edges] edge index
            measurements: Optional measurement dict for power injection data

        Returns:
            loss_dict: Dictionary of loss values at each level.
        """
        corrected_states, bus_loss = self.bus_physics(
            states=states,
            parameters=parameters,
            edge_index=edge_index,
            measurements=measurements,
        )
        physics_states = prepare_states_for_physics(corrected_states, measurements)

        if self.hierarchical_enabled:
            feeder_loss = self._compute_feeder_level_loss(physics_states)
            substation_loss = self._compute_substation_level_loss(physics_states)
        else:
            feeder_loss = torch.tensor(0.0, device=states["v_mag"].device)
            substation_loss = torch.tensor(0.0, device=states["v_mag"].device)

        total_loss = (
            self.bus_weight * bus_loss
            + self.feeder_weight * feeder_loss
            + self.substation_weight * substation_loss
        )

        total_loss = torch.clamp(total_loss, max=1000.0)

        return {
            "bus": bus_loss,
            "feeder": feeder_loss,
            "substation": substation_loss,
            "total": total_loss,
        }

    def _compute_feeder_level_loss(
        self, states: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute feeder-level power balance constraints.

        For each feeder, the voltage variance should be small
        (feeders are local, voltages should be similar).
        """
        if "p_bus" not in states:
            return torch.tensor(0.0, device=states["v_mag"].device)

        feeder_losses = []
        num_nodes = states["v_mag"].shape[1]
        for feeder_id in range(self.num_feeders):
            buses = self.feeder_assignment.get_feeder_buses(feeder_id)
            if len(buses) == 0:
                continue

            valid_buses = [bus for bus in buses if 0 <= bus < num_nodes]
            if len(valid_buses) == 0:
                continue

            bus_idx = torch.tensor(
                valid_buses, dtype=torch.long, device=states["v_mag"].device
            )

            v_feeder = states["v_mag"][:, bus_idx]
            v_variance = v_feeder.var(dim=1).mean()

            feeder_losses.append(v_variance)

        if len(feeder_losses) == 0:
            return torch.tensor(0.0, device=states["v_mag"].device)

        return torch.stack(feeder_losses).mean()

    def _compute_substation_level_loss(
        self, states: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute substation-level overall power balance constraint.

        Sum of all bus injections should be balanced.
        """
        if "p_bus" not in states:
            return torch.tensor(0.0, device=states["v_mag"].device)

        total_p = states["p_bus"].sum(dim=1)
        total_q = states["q_bus"].sum(dim=1)

        return (total_p**2 + total_q**2).mean()


if __name__ == "__main__":
    hpc = HierarchicalPhysicsConstraints(num_feeders=3)
    states = {
        "v_mag": torch.rand(2, 123) * 0.1 + 0.95,
        "v_ang": torch.randn(2, 123) * 0.05,
        "p_bus": torch.randn(2, 123) * 0.005,
        "q_bus": torch.randn(2, 123) * 0.005,
    }
    parameters = {
        "r_line": torch.rand(2, 240) * 0.5,
        "x_line": torch.rand(2, 240) * 0.2,
    }
    edge_index = torch.randint(0, 123, (2, 240))

    losses = hpc(states, parameters, edge_index)
    for k, v in losses.items():
        val = v.item() if isinstance(v, torch.Tensor) else v
        print("  %s: %.6f" % (k, val))
