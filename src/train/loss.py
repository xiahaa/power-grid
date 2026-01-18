"""
Loss functions for joint state and parameter estimation

Author: Your Name
Date: 2026-01-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class JointEstimationLoss(nn.Module):
    """
    Combined loss for joint state and parameter estimation

    L = w_state * L_state + w_param * L_param + w_physics * L_physics + w_smooth * L_smooth
    """

    def __init__(
        self,
        state_weight: float = 1.0,
        parameter_weight: float = 0.5,
        physics_weight: float = 0.1,
        temporal_smoothness: float = 0.01
    ):
        super().__init__()

        self.state_weight = state_weight
        self.parameter_weight = parameter_weight
        self.physics_weight = physics_weight
        self.temporal_smoothness = temporal_smoothness

    def state_loss(
        self,
        pred_states: Dict[str, torch.Tensor],
        true_states: Dict[str, torch.Tensor],
        obs_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        State estimation loss (voltage magnitude and angle)

        Args:
            pred_states: Dict with 'v_mag', 'v_ang'
            true_states: Dict with 'v_mag', 'v_ang'
            obs_mask: Optional mask for observed buses
        """
        loss = 0.0

        for key in ['v_mag', 'v_ang']:
            pred = pred_states[key]
            target = true_states[key]

            # MSE loss
            if obs_mask is not None:
                # Weight observed buses more
                loss_all = F.mse_loss(pred, target, reduction='none')
                # Expand obs_mask to match dimensions if needed
                if obs_mask.dim() < pred.dim():
                    obs_mask = obs_mask.unsqueeze(1)  # Add time dimension
                weighted_loss = loss_all * (obs_mask.float() * 2.0 + 0.5)
                loss += weighted_loss.mean()
            else:
                loss += F.mse_loss(pred, target)

        return loss

    def parameter_loss(
        self,
        pred_params: Dict[str, torch.Tensor],
        true_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Parameter estimation loss (line impedance)

        Args:
            pred_params: Dict with 'r_line', 'x_line'
            true_params: Dict with 'r_line', 'x_line'
        """
        loss = 0.0

        for key in ['r_line', 'x_line']:
            pred = pred_params[key]
            target = true_params[key]

            # MSE loss
            loss += F.mse_loss(pred, target)

            # Relative error (percentage)
            rel_error = torch.abs((pred - target) / (target + 1e-8))
            loss += rel_error.mean() * 0.1

        return loss

    def temporal_smoothness_loss(
        self,
        states_sequence: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Temporal smoothness regularization

        Encourages smooth transitions between time steps

        Args:
            states_sequence: Dict with temporal sequences
        """
        if 'v_mag' not in states_sequence:
            return torch.tensor(0.0)

        loss = 0.0
        for key, values in states_sequence.items():
            if values.dim() >= 2:  # Has temporal dimension
                # Compute differences between consecutive time steps
                diff = values[:, 1:] - values[:, :-1]
                loss += (diff ** 2).mean()

        return loss

    def forward(
        self,
        pred_states: Dict[str, torch.Tensor],
        true_states: Dict[str, torch.Tensor],
        pred_params: Dict[str, torch.Tensor],
        true_params: Dict[str, torch.Tensor],
        physics_loss: torch.Tensor,
        obs_mask: torch.Tensor = None,
        states_sequence: Dict[str, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss

        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary of individual loss components
        """
        # Individual losses
        l_state = self.state_loss(pred_states, true_states, obs_mask)
        l_param = self.parameter_loss(pred_params, true_params)
        l_physics = physics_loss
        l_smooth = (
            self.temporal_smoothness_loss(states_sequence)
            if states_sequence is not None else torch.tensor(0.0)
        )

        # Weighted sum
        total_loss = (
            self.state_weight * l_state +
            self.parameter_weight * l_param +
            self.physics_weight * l_physics +
            self.temporal_smoothness * l_smooth
        )

        # Return loss components for logging
        loss_dict = {
            'total': total_loss.item(),
            'state': l_state.item(),
            'parameter': l_param.item(),
            'physics': l_physics.item(),
            'smoothness': l_smooth.item() if isinstance(l_smooth, torch.Tensor) else l_smooth
        }

        return total_loss, loss_dict


if __name__ == "__main__":
    from typing import Tuple

    # Test loss
    batch_size, num_nodes, num_edges = 4, 33, 32

    pred_states = {
        'v_mag': torch.rand(batch_size, num_nodes),
        'v_ang': torch.randn(batch_size, num_nodes) * 0.1
    }

    true_states = {
        'v_mag': torch.rand(batch_size, num_nodes),
        'v_ang': torch.randn(batch_size, num_nodes) * 0.1
    }

    pred_params = {
        'r_line': torch.rand(batch_size, num_edges),
        'x_line': torch.rand(batch_size, num_edges)
    }

    true_params = {
        'r_line': torch.rand(batch_size, num_edges),
        'x_line': torch.rand(batch_size, num_edges)
    }

    physics_loss = torch.tensor(0.05)

    criterion = JointEstimationLoss()
    total_loss, loss_dict = criterion(
        pred_states, true_states,
        pred_params, true_params,
        physics_loss
    )

    print("Loss components:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.6f}")
