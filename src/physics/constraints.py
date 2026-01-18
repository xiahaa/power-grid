"""
Physics-Informed Constraints for Power Grid Estimation

Implements differentiable optimization layer to enforce:
1. Power Flow Equations (KCL/KVL)
2. Voltage Limits
3. Manifold Projection

Two modes:
- Soft constraints: Add penalty terms to loss
- Hard constraints: Project to feasible manifold via differentiable optimization

Author: Your Name
Date: 2026-01-18
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import warnings


class PowerFlowConstraints:
    """Encodes AC power flow equations as differentiable constraints"""

    @staticmethod
    def compute_power_mismatch(
        v_mag: torch.Tensor,
        v_ang: torch.Tensor,
        edge_index: torch.Tensor,
        r_line: torch.Tensor,
        x_line: torch.Tensor,
        p_inj: Optional[torch.Tensor] = None,
        q_inj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute power flow mismatch for KCL equations

        AC Power Flow:
            P_i = V_i * Σ_j V_j * (G_ij*cos(θ_i - θ_j) + B_ij*sin(θ_i - θ_j))
            Q_i = V_i * Σ_j V_j * (G_ij*sin(θ_i - θ_j) - B_ij*cos(θ_i - θ_j))

        Args:
            v_mag: Voltage magnitudes [batch_size, num_nodes]
            v_ang: Voltage angles [batch_size, num_nodes]
            edge_index: [2, num_edges]
            r_line: Line resistance [batch_size, num_edges]
            x_line: Line reactance [batch_size, num_edges]
            p_inj: Active power injection [batch_size, num_nodes]
            q_inj: Reactive power injection [batch_size, num_nodes]

        Returns:
            p_mismatch: [batch_size, num_nodes]
            q_mismatch: [batch_size, num_nodes]
        """
        batch_size, num_nodes = v_mag.shape
        num_edges = edge_index.shape[1]

        # Compute admittance matrix elements
        z_squared = r_line ** 2 + x_line ** 2
        g_line = r_line / z_squared  # Conductance
        b_line = -x_line / z_squared  # Susceptance

        # Initialize calculated power injections
        p_calc = torch.zeros_like(v_mag)
        q_calc = torch.zeros_like(v_mag)

        # Iterate over edges to compute power flows
        from_bus = edge_index[0]
        to_bus = edge_index[1]

        for edge_idx in range(num_edges):
            i = from_bus[edge_idx]
            j = to_bus[edge_idx]

            # Voltage products
            v_i = v_mag[:, i]
            v_j = v_mag[:, j]
            theta_ij = v_ang[:, i] - v_ang[:, j]

            # Admittance for this line
            g_ij = g_line[:, edge_idx]
            b_ij = b_line[:, edge_idx]

            # Power flow from i to j
            p_ij = v_i * v_j * (g_ij * torch.cos(theta_ij) + b_ij * torch.sin(theta_ij))
            q_ij = v_i * v_j * (g_ij * torch.sin(theta_ij) - b_ij * torch.cos(theta_ij))

            # Accumulate at bus i (outgoing)
            p_calc[:, i] = p_calc[:, i] + p_ij
            q_calc[:, i] = q_calc[:, i] + q_ij

        # Mismatch = Calculated - Injected
        if p_inj is not None and q_inj is not None:
            p_mismatch = p_calc - p_inj
            q_mismatch = q_calc - q_inj
        else:
            # If no injection provided, just return calculated
            p_mismatch = p_calc
            q_mismatch = q_calc

        return p_mismatch, q_mismatch

    @staticmethod
    def voltage_limit_penalty(
        v_mag: torch.Tensor,
        v_min: float = 0.95,
        v_max: float = 1.05
    ) -> torch.Tensor:
        """
        Penalty for voltage violations

        Args:
            v_mag: [batch_size, num_nodes]
            v_min, v_max: Voltage limits in p.u.

        Returns:
            penalty: [batch_size]
        """
        lower_violation = F.relu(v_min - v_mag)
        upper_violation = F.relu(v_mag - v_max)
        penalty = (lower_violation ** 2 + upper_violation ** 2).sum(dim=-1)
        return penalty


class PhysicsInformedLayer(nn.Module):
    """
    Differentiable physics layer that enforces power flow constraints

    Two modes:
    1. Soft: Add constraint violations as penalty to loss
    2. Hard: Project outputs to feasible manifold
    """

    def __init__(
        self,
        constraint_type: str = "soft",
        projection_method: str = "gradient_descent",
        max_iterations: int = 20,
        tolerance: float = 1e-4,
        voltage_limits: Tuple[float, float] = (0.95, 1.05),
        power_balance_weight: float = 10.0
    ):
        """
        Args:
            constraint_type: "soft" or "hard"
            projection_method: "gradient_descent" or "cvxpy" (for hard)
            max_iterations: Max iterations for projection
            tolerance: Convergence tolerance
            voltage_limits: (v_min, v_max) in p.u.
            power_balance_weight: Weight for power balance constraint
        """
        super().__init__()

        self.constraint_type = constraint_type
        self.projection_method = projection_method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.v_min, self.v_max = voltage_limits
        self.power_balance_weight = power_balance_weight

        self.power_flow = PowerFlowConstraints()

    def forward(
        self,
        states: Dict[str, torch.Tensor],
        parameters: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        measurements: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Apply physics constraints to model outputs

        Args:
            states: Dict with 'v_mag', 'v_ang'
            parameters: Dict with 'r_line', 'x_line'
            edge_index: [2, num_edges]
            measurements: Optional dict with 'p_bus', 'q_bus' for mismatch calculation

        Returns:
            corrected_states: Physically consistent states
            constraint_loss: Constraint violation (for soft mode)
        """
        v_mag = states['v_mag']
        v_ang = states['v_ang']
        r_line = parameters['r_line']
        x_line = parameters['x_line']

        if self.constraint_type == "soft":
            return self._soft_constraints(
                v_mag, v_ang, r_line, x_line, edge_index, measurements
            )
        else:  # hard
            return self._hard_constraints(
                v_mag, v_ang, r_line, x_line, edge_index, measurements
            )

    def _soft_constraints(
        self,
        v_mag: torch.Tensor,
        v_ang: torch.Tensor,
        r_line: torch.Tensor,
        x_line: torch.Tensor,
        edge_index: torch.Tensor,
        measurements: Optional[Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Compute constraint violation as penalty"""

        # Power flow mismatch
        p_inj = measurements.get('p_bus') if measurements else None
        q_inj = measurements.get('q_bus') if measurements else None

        p_mismatch, q_mismatch = self.power_flow.compute_power_mismatch(
            v_mag, v_ang, edge_index, r_line, x_line, p_inj, q_inj
        )

        # L2 norm of power mismatch
        power_loss = (p_mismatch ** 2 + q_mismatch ** 2).sum(dim=-1).mean()

        # Voltage limit penalty
        voltage_loss = self.power_flow.voltage_limit_penalty(
            v_mag, self.v_min, self.v_max
        ).mean()

        # Total constraint loss
        constraint_loss = self.power_balance_weight * power_loss + voltage_loss

        # Return original states (no correction in soft mode)
        states = {'v_mag': v_mag, 'v_ang': v_ang}
        return states, constraint_loss

    def _hard_constraints(
        self,
        v_mag: torch.Tensor,
        v_ang: torch.Tensor,
        r_line: torch.Tensor,
        x_line: torch.Tensor,
        edge_index: torch.Tensor,
        measurements: Optional[Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Project to feasible manifold via gradient descent"""

        if self.projection_method == "gradient_descent":
            return self._project_gradient_descent(
                v_mag, v_ang, r_line, x_line, edge_index, measurements
            )
        else:
            warnings.warn(f"Projection method {self.projection_method} not implemented, using soft")
            return self._soft_constraints(
                v_mag, v_ang, r_line, x_line, edge_index, measurements
            )

    def _project_gradient_descent(
        self,
        v_mag: torch.Tensor,
        v_ang: torch.Tensor,
        r_line: torch.Tensor,
        x_line: torch.Tensor,
        edge_index: torch.Tensor,
        measurements: Optional[Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Project to manifold using differentiable gradient descent

        Iteratively minimize constraint violations while maintaining gradients
        """
        # Make copies for projection
        v_mag_proj = v_mag.clone().requires_grad_(True)
        v_ang_proj = v_ang.clone().requires_grad_(True)

        # Projection optimizer
        optimizer = torch.optim.LBFGS(
            [v_mag_proj, v_ang_proj],
            lr=0.1,
            max_iter=self.max_iterations,
            tolerance_grad=self.tolerance,
            tolerance_change=self.tolerance,
            line_search_fn='strong_wolfe'
        )

        def closure():
            optimizer.zero_grad()

            # Compute constraint violations
            p_inj = measurements.get('p_bus') if measurements else None
            q_inj = measurements.get('q_bus') if measurements else None

            p_mismatch, q_mismatch = self.power_flow.compute_power_mismatch(
                v_mag_proj, v_ang_proj, edge_index, r_line, x_line, p_inj, q_inj
            )

            power_loss = (p_mismatch ** 2 + q_mismatch ** 2).sum(dim=-1).mean()
            voltage_loss = self.power_flow.voltage_limit_penalty(
                v_mag_proj, self.v_min, self.v_max
            ).mean()

            # Proximity to original prediction (regularization)
            proximity_loss = (
                ((v_mag_proj - v_mag) ** 2).mean() +
                ((v_ang_proj - v_ang) ** 2).mean()
            )

            loss = (
                self.power_balance_weight * power_loss +
                voltage_loss +
                0.1 * proximity_loss  # Don't deviate too much
            )

            loss.backward()
            return loss

        # Run optimization
        optimizer.step(closure)

        # Enforce voltage limits explicitly
        v_mag_proj = torch.clamp(v_mag_proj, self.v_min, self.v_max)

        # Compute final constraint loss (for monitoring)
        with torch.no_grad():
            p_inj = measurements.get('p_bus') if measurements else None
            q_inj = measurements.get('q_bus') if measurements else None

            p_mismatch, q_mismatch = self.power_flow.compute_power_mismatch(
                v_mag_proj, v_ang_proj, edge_index, r_line, x_line, p_inj, q_inj
            )
            constraint_loss = (p_mismatch ** 2 + q_mismatch ** 2).sum(dim=-1).mean()

        states = {'v_mag': v_mag_proj, 'v_ang': v_ang_proj}
        return states, constraint_loss


class PhysicsInformedGraphMamba(nn.Module):
    """Graph Mamba + Physics Constraints (End-to-end model)"""

    def __init__(
        self,
        graph_mamba: nn.Module,
        physics_layer: PhysicsInformedLayer
    ):
        super().__init__()
        self.graph_mamba = graph_mamba
        self.physics_layer = physics_layer

    def forward(
        self,
        measurements: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass with physics correction

        Returns:
            states: Corrected states
            parameters: Estimated parameters
            constraint_loss: Physics constraint violation
        """
        # Model prediction
        states, parameters = self.graph_mamba(
            measurements, edge_index, edge_attr, obs_mask
        )

        # Apply physics constraints
        corrected_states, constraint_loss = self.physics_layer(
            states, parameters, edge_index, measurements
        )

        return corrected_states, parameters, constraint_loss


if __name__ == "__main__":
    import torch.nn.functional as F

    # Test physics constraints
    batch_size, num_nodes, num_edges = 4, 33, 32

    v_mag = torch.rand(batch_size, num_nodes) * 0.2 + 0.9  # [0.9, 1.1]
    v_ang = torch.randn(batch_size, num_nodes) * 0.1
    r_line = torch.rand(batch_size, num_edges) * 0.5
    x_line = torch.rand(batch_size, num_edges) * 0.5
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Test soft constraints
    physics_layer = PhysicsInformedLayer(constraint_type="soft")
    states = {'v_mag': v_mag, 'v_ang': v_ang}
    parameters = {'r_line': r_line, 'x_line': x_line}

    corrected_states, loss = physics_layer(states, parameters, edge_index)

    print(f"Constraint loss: {loss.item():.6f}")
    print(f"Voltage range: [{corrected_states['v_mag'].min():.3f}, {corrected_states['v_mag'].max():.3f}]")
