"""
Graph Mamba Architecture for Power Grid Estimation

Spatial-Temporal Decoupled Design:
1. Spatial Encoder: GAT for topology-aware feature extraction
2. Temporal Encoder: Mamba for long-term dependency modeling
3. Dual Heads: State estimation + Parameter estimation

Author: Your Name
Date: 2026-01-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphSAGE
from mamba_ssm import Mamba
from typing import Dict, Tuple, Optional


class SpatialEncoder(nn.Module):
    """Graph Neural Network for spatial feature extraction"""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        encoder_type: str = "GAT"
    ):
        """
        Args:
            in_channels: Input feature dimension per node
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads (for GAT)
            num_layers: Number of GNN layers
            dropout: Dropout rate
            encoder_type: "GAT" or "GraphSage"
        """
        super().__init__()

        self.encoder_type = encoder_type
        self.num_layers = num_layers

        if encoder_type == "GAT":
            self.convs = nn.ModuleList()
            for i in range(num_layers):
                in_dim = in_channels if i == 0 else hidden_dim * num_heads
                out_dim = hidden_dim
                self.convs.append(
                    GATConv(
                        in_dim,
                        out_dim,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True
                    )
                )
        else:
            raise NotImplementedError(f"Encoder {encoder_type} not implemented")

        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * num_heads

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [batch_size, num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]

        Returns:
            h: Spatial features [batch_size, num_nodes, output_dim]
        """
        batch_size, num_nodes, _ = x.shape

        # Process each graph in batch
        h_list = []
        for b in range(batch_size):
            h = x[b]  # [num_nodes, in_channels]

            for conv in self.convs:
                h = conv(h, edge_index, edge_attr=edge_attr)
                h = F.elu(h)
                h = self.dropout(h)

            h_list.append(h)

        h = torch.stack(h_list, dim=0)  # [batch_size, num_nodes, output_dim]
        return h


class MambaBlock(nn.Module):
    """Mamba SSM for temporal sequence modeling"""

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 3
    ):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Local convolution width
            expand: Expansion factor
            num_layers: Number of Mamba layers
        """
        super().__init__()

        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence [batch_size, seq_len, d_model]

        Returns:
            h: Output sequence [batch_size, seq_len, d_model]
        """
        h = x
        for layer in self.layers:
            h = h + layer(self.norm(h))  # Residual connection
        return h


class StateHead(nn.Module):
    """Output head for state estimation (voltage magnitude & angle)"""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list = [128, 64],
        output_dim: int = 2,  # V_mag, V_ang per bus
        activation: str = "relu"
    ):
        super().__init__()

        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input features [batch_size, num_nodes, in_dim]

        Returns:
            states: Dict with 'v_mag' and 'v_ang'
        """
        out = self.network(x)  # [batch_size, num_nodes, 2]

        # Split into magnitude and angle
        v_mag = torch.sigmoid(out[..., 0]) * 0.3 + 0.85  # Constrain to [0.85, 1.15]
        v_ang = torch.tanh(out[..., 1]) * 0.5  # Constrain to [-0.5, 0.5] rad

        return {
            'v_mag': v_mag,
            'v_ang': v_ang
        }


class ParameterHead(nn.Module):
    """Output head for parameter estimation (line R, X)"""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list = [128, 64],
        output_dim: int = 2,  # R, X per line
        temporal_pooling: str = "ewma",
        alpha: float = 0.9
    ):
        super().__init__()

        self.temporal_pooling = temporal_pooling
        self.alpha = alpha

        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Edge pooling: aggregate node features to edges
        self.edge_pooling = nn.Linear(in_dim * 2, in_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_features: [batch_size, seq_len, num_nodes, in_dim]
            edge_index: [2, num_edges]

        Returns:
            parameters: Dict with 'r_line' and 'x_line'
        """
        batch_size, seq_len, num_nodes, in_dim = node_features.shape
        num_edges = edge_index.shape[1]

        # Pool node features to edge features
        from_nodes = edge_index[0]
        to_nodes = edge_index[1]

        edge_features = []
        for b in range(batch_size):
            batch_edge_feat = []
            for t in range(seq_len):
                from_feat = node_features[b, t, from_nodes]  # [num_edges, in_dim]
                to_feat = node_features[b, t, to_nodes]
                concat_feat = torch.cat([from_feat, to_feat], dim=-1)  # [num_edges, 2*in_dim]
                edge_feat = self.edge_pooling(concat_feat)  # [num_edges, in_dim]
                batch_edge_feat.append(edge_feat)
            edge_features.append(torch.stack(batch_edge_feat))

        edge_features = torch.stack(edge_features)  # [batch_size, seq_len, num_edges, in_dim]

        # Temporal pooling (parameters change slowly)
        if self.temporal_pooling == "mean":
            pooled = edge_features.mean(dim=1)  # [batch_size, num_edges, in_dim]
        elif self.temporal_pooling == "ewma":
            # Exponential weighted moving average (more weight on recent)
            weights = torch.tensor(
                [self.alpha ** (seq_len - 1 - t) for t in range(seq_len)],
                device=edge_features.device
            )
            weights = weights / weights.sum()
            pooled = (edge_features * weights.view(1, -1, 1, 1)).sum(dim=1)
        else:
            pooled = edge_features[:, -1]  # Last time step

        # Predict parameters
        out = self.network(pooled)  # [batch_size, num_edges, 2]

        # Ensure positive parameters
        r_line = F.softplus(out[..., 0])
        x_line = F.softplus(out[..., 1])

        return {
            'r_line': r_line,
            'x_line': x_line
        }


class GraphMamba(nn.Module):
    """
    Physics-Informed Graph Mamba for Joint State & Parameter Estimation

    Architecture:
        Input → Spatial Encoder (GAT) → Temporal Encoder (Mamba) → Dual Heads
    """

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        input_dim: int = 3,  # v_mag, p_bus, q_bus
        spatial_config: dict = None,
        temporal_config: dict = None,
        state_head_config: dict = None,
        parameter_head_config: dict = None
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_edges

        # Default configs
        spatial_config = spatial_config or {
            'hidden_dim': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1
        }

        temporal_config = temporal_config or {
            'd_model': 64,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'num_layers': 3
        }

        # Input projection
        self.input_proj = nn.Linear(input_dim, spatial_config['hidden_dim'])

        # Spatial encoder - map config keys to expected parameters
        spatial_encoder_kwargs = spatial_config.copy()
        if 'type' in spatial_encoder_kwargs:
            spatial_encoder_kwargs['encoder_type'] = spatial_encoder_kwargs.pop('type')
        self.spatial_encoder = SpatialEncoder(
            in_channels=spatial_config['hidden_dim'],
            **spatial_encoder_kwargs
        )

        spatial_out_dim = self.spatial_encoder.output_dim

        # Temporal encoder (node-wise)
        temporal_config['d_model'] = spatial_out_dim
        # Remove type key as MambaBlock doesn't expect it
        temporal_encoder_kwargs = {k: v for k, v in temporal_config.items() if k != 'type'}
        self.temporal_encoder = MambaBlock(**temporal_encoder_kwargs)

        # Output heads
        self.state_head = StateHead(
            in_dim=spatial_out_dim,
            **(state_head_config or {})
        )

        self.parameter_head = ParameterHead(
            in_dim=spatial_out_dim,
            **(parameter_head_config or {})
        )

    def forward(
        self,
        measurements: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            measurements: Dict with keys 'v_mag', 'p_bus', 'q_bus'
                         Each: [batch_size, seq_len, num_nodes]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_dim]
            obs_mask: [batch_size, seq_len, num_nodes] (1=observed, 0=missing)

        Returns:
            states: Dict with estimated states
            parameters: Dict with estimated parameters
        """
        batch_size, seq_len, num_nodes = measurements['v_mag'].shape

        # Stack measurements as node features
        x = torch.stack([
            measurements['v_mag'],
            measurements['p_bus'],
            measurements['q_bus']
        ], dim=-1)  # [batch_size, seq_len, num_nodes, 3]

        # Handle missing measurements
        if obs_mask is not None:
            x = x * obs_mask.unsqueeze(-1)

        # Input projection
        x = self.input_proj(x)  # [batch_size, seq_len, num_nodes, hidden_dim]

        # Spatial encoding (per time step)
        spatial_features = []
        for t in range(seq_len):
            h_spatial = self.spatial_encoder(
                x[:, t], edge_index, edge_attr
            )  # [batch_size, num_nodes, spatial_dim]
            spatial_features.append(h_spatial)

        spatial_features = torch.stack(spatial_features, dim=1)
        # [batch_size, seq_len, num_nodes, spatial_dim]

        # Temporal encoding (per node)
        batch_size, seq_len, num_nodes, spatial_dim = spatial_features.shape
        temporal_features = []
        for n in range(num_nodes):
            node_seq = spatial_features[:, :, n, :]  # [batch_size, seq_len, spatial_dim]
            h_temporal = self.temporal_encoder(node_seq)  # [batch_size, seq_len, spatial_dim]
            temporal_features.append(h_temporal)

        temporal_features = torch.stack(temporal_features, dim=2)
        # [batch_size, seq_len, num_nodes, spatial_dim]

        # State estimation (use last time step for prediction)
        states = self.state_head(temporal_features[:, -1])
        # Returns dict: {'v_mag': [batch_size, num_nodes], 'v_ang': [...]}

        # Parameter estimation (use full sequence with temporal pooling)
        parameters = self.parameter_head(temporal_features, edge_index)
        # Returns dict: {'r_line': [batch_size, num_edges], 'x_line': [...]}

        return states, parameters


if __name__ == "__main__":
    # Test model
    model = GraphMamba(
        num_nodes=33,
        num_edges=32,
        input_dim=3
    )

    # Dummy data
    batch_size, seq_len = 4, 10
    measurements = {
        'v_mag': torch.randn(batch_size, seq_len, 33),
        'p_bus': torch.randn(batch_size, seq_len, 33),
        'q_bus': torch.randn(batch_size, seq_len, 33),
    }
    edge_index = torch.randint(0, 33, (2, 64))  # Bidirectional

    states, parameters = model(measurements, edge_index)

    print("Output shapes:")
    print(f"  V_mag: {states['v_mag'].shape}")
    print(f"  V_ang: {states['v_ang'].shape}")
    print(f"  R_line: {parameters['r_line'].shape}")
    print(f"  X_line: {parameters['x_line'].shape}")
