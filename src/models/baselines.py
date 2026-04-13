"""
Baseline DSSE models for comparison with MultiRateMambaFusion.

Provides four baseline estimators covering different combinations of
spatial and temporal processing:

1. WLSEstimator    - MLP-only, no temporal, no spatial
2. GNNEstimator    - GAT spatial, no temporal
3. LSTMEstimator   - LSTM temporal, no spatial
4. TransformerEstimator - Transformer temporal, no spatial

All baselines share a common interface:
    forward(measurements, edge_index, edge_attr, obs_mask)
        -> (states_dict, params_dict)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class WLSEstimator(nn.Module):
    """
    Weighted Least Squares baseline for DSSE.

    Uses a two-layer MLP to map flattened last-timestep measurements
    directly to state and parameter estimates. No temporal or spatial
    processing.

    Args:
        num_nodes: Number of buses in the network.
        num_edges: Number of edges (lines) in the network.
        input_dim: Feature dimension per node (default 3).
        hidden_dim: Hidden dimension for the MLP (default 64).
    """

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        input_dim: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        self.state_mlp = nn.Sequential(
            nn.Linear(num_nodes * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes * 2),
        )

        self.param_mlp = nn.Sequential(
            nn.Linear(num_nodes * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_edges * 2),
        )

    def forward(
        self,
        measurements: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            measurements: Dict with 'v_mag', 'p_bus', 'q_bus' each of
                shape [batch, seq_len, num_nodes].
            edge_index: Shape [2, num_edges].
            edge_attr: Optional, shape [num_edges, edge_dim].
            obs_mask: Optional, shape [batch, seq_len, num_nodes].

        Returns:
            Tuple of (states, params) where states has 'v_mag' and
            'v_ang' of shape [batch, num_nodes], and params has
            'r_line' and 'x_line' of shape [batch, num_edges].
        """
        v_mag = torch.nan_to_num(measurements["v_mag"][:, -1])
        p_bus = torch.nan_to_num(measurements["p_bus"][:, -1])
        q_bus = torch.nan_to_num(measurements["q_bus"][:, -1])

        x = torch.stack([v_mag, p_bus, q_bus], dim=-1)
        if obs_mask is not None:
            x = x * obs_mask[:, -1].unsqueeze(-1).float()
        x_flat = x.reshape(x.size(0), -1)

        state_out = self.state_mlp(x_flat)
        state_out = state_out.reshape(x.size(0), self.num_nodes, 2)
        v_mag_pred = torch.sigmoid(state_out[..., 0]) * 0.3 + 0.85
        v_ang_pred = torch.tanh(state_out[..., 1]) * 0.5
        states = {"v_mag": v_mag_pred, "v_ang": v_ang_pred}

        param_out = self.param_mlp(x_flat)
        param_out = param_out.reshape(x.size(0), self.num_edges, 2)
        r_line = torch.sigmoid(param_out[..., 0]) * 2.99 + 0.01
        x_line = torch.sigmoid(param_out[..., 1]) * 2.99 + 0.01
        params = {"r_line": r_line, "x_line": x_line}

        return states, params


class GNNEstimator(nn.Module):
    """
    GNN-only baseline for DSSE.

    Uses two GAT layers for spatial feature extraction followed by
    MLP heads for state and parameter estimation. No temporal
    processing. Takes only the last timestep of measurements.

    Args:
        num_nodes: Number of buses in the network.
        num_edges: Number of edges (lines) in the network.
        input_dim: Feature dimension per node (default 3).
        hidden_dim: GAT hidden dimension per head (default 128).
        num_heads: Number of attention heads (default 4).
    """

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        from torch_geometric.nn import GATConv

        self.out_dim = hidden_dim * num_heads

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gat1 = GATConv(
            hidden_dim, hidden_dim, heads=num_heads, dropout=0.1, concat=True
        )
        self.gat2 = GATConv(
            self.out_dim, hidden_dim, heads=num_heads, dropout=0.1, concat=True
        )
        self.dropout = nn.Dropout(0.1)

        self.state_head = nn.Sequential(
            nn.Linear(self.out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        self.param_edge_pool = nn.Linear(self.out_dim * 2, self.out_dim)
        self.param_head = nn.Sequential(
            nn.Linear(self.out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        measurements: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            measurements: Dict with 'v_mag', 'p_bus', 'q_bus' each of
                shape [batch, seq_len, num_nodes].
            edge_index: Shape [2, num_edges].
            edge_attr: Optional, shape [num_edges, edge_dim].
            obs_mask: Optional, shape [batch, seq_len, num_nodes].

        Returns:
            Tuple of (states, params) where states has 'v_mag' and
            'v_ang' of shape [batch, num_nodes], and params has
            'r_line' and 'x_line' of shape [batch, num_edges].
        """
        v_mag = torch.nan_to_num(measurements["v_mag"][:, -1])
        p_bus = torch.nan_to_num(measurements["p_bus"][:, -1])
        q_bus = torch.nan_to_num(measurements["q_bus"][:, -1])

        x = torch.stack([v_mag, p_bus, q_bus], dim=-1)
        if obs_mask is not None:
            x = x * obs_mask[:, -1].unsqueeze(-1).float()

        x = self.input_proj(x)
        batch_size = x.size(0)

        h_list = []
        for b in range(batch_size):
            h = self.gat1(x[b], edge_index, edge_attr=edge_attr)
            h = F.relu(h)
            h = self.dropout(h)
            h = self.gat2(h, edge_index, edge_attr=edge_attr)
            h = F.relu(h)
            h = self.dropout(h)
            h_list.append(h)
        h = torch.stack(h_list, dim=0)

        state_out = self.state_head(h)
        v_mag_pred = torch.sigmoid(state_out[..., 0]) * 0.3 + 0.85
        v_ang_pred = torch.tanh(state_out[..., 1]) * 0.5
        states = {"v_mag": v_mag_pred, "v_ang": v_ang_pred}

        from_nodes = edge_index[0]
        to_nodes = edge_index[1]
        edge_feat_list = []
        for b in range(batch_size):
            from_feat = h[b, from_nodes]
            to_feat = h[b, to_nodes]
            edge_feat = self.param_edge_pool(torch.cat([from_feat, to_feat], dim=-1))
            edge_feat_list.append(edge_feat)
        edge_feat = torch.stack(edge_feat_list, dim=0)

        param_out = self.param_head(edge_feat)
        r_line = torch.sigmoid(param_out[..., 0]) * 2.99 + 0.01
        x_line = torch.sigmoid(param_out[..., 1]) * 2.99 + 0.01
        params = {"r_line": r_line, "x_line": x_line}

        return states, params


class LSTMEstimator(nn.Module):
    """
    LSTM-only baseline for DSSE.

    Flattens the node dimension and processes the full temporal sequence
    with a 2-layer LSTM. The last hidden state is fed to MLP heads for
    state and parameter estimation. No spatial processing.

    Args:
        num_nodes: Number of buses in the network.
        num_edges: Number of edges (lines) in the network.
        input_dim: Feature dimension per node (default 3).
        hidden_dim: LSTM hidden dimension (default 128).
    """

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        input_dim: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        self.lstm = nn.LSTM(
            input_size=num_nodes * input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_nodes * 2),
        )

        self.param_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_edges * 2),
        )

    def forward(
        self,
        measurements: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            measurements: Dict with 'v_mag', 'p_bus', 'q_bus' each of
                shape [batch, seq_len, num_nodes].
            edge_index: Shape [2, num_edges].
            edge_attr: Optional, shape [num_edges, edge_dim].
            obs_mask: Optional, shape [batch, seq_len, num_nodes].

        Returns:
            Tuple of (states, params) where states has 'v_mag' and
            'v_ang' of shape [batch, num_nodes], and params has
            'r_line' and 'x_line' of shape [batch, num_edges].
        """
        v_mag = torch.nan_to_num(measurements["v_mag"])
        p_bus = torch.nan_to_num(measurements["p_bus"])
        q_bus = torch.nan_to_num(measurements["q_bus"])

        x = torch.stack([v_mag, p_bus, q_bus], dim=-1)
        if obs_mask is not None:
            x = x * obs_mask.unsqueeze(-1).float()

        batch_size = x.size(0)
        seq_len = x.size(1)
        x_flat = x.reshape(batch_size, seq_len, -1)

        _, (h_n, _) = self.lstm(x_flat)
        h_last = h_n[-1]

        state_out = self.state_head(h_last)
        state_out = state_out.reshape(batch_size, self.num_nodes, 2)
        v_mag_pred = torch.sigmoid(state_out[..., 0]) * 0.3 + 0.85
        v_ang_pred = torch.tanh(state_out[..., 1]) * 0.5
        states = {"v_mag": v_mag_pred, "v_ang": v_ang_pred}

        param_out = self.param_head(h_last)
        param_out = param_out.reshape(batch_size, self.num_edges, 2)
        r_line = torch.sigmoid(param_out[..., 0]) * 2.99 + 0.01
        x_line = torch.sigmoid(param_out[..., 1]) * 2.99 + 0.01
        params = {"r_line": r_line, "x_line": x_line}

        return states, params


class TransformerEstimator(nn.Module):
    """
    Transformer-only baseline for DSSE.

    Uses a multi-head attention encoder over the sequence dimension
    followed by mean pooling and MLP heads. No spatial processing.

    Args:
        num_nodes: Number of buses in the network.
        num_edges: Number of edges (lines) in the network.
        input_dim: Feature dimension per node (default 3).
        d_model: Transformer model dimension (default 128).
        num_heads: Number of attention heads (default 4).
        num_layers: Number of encoder layers (default 2).
    """

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        input_dim: int = 3,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        self.input_proj = nn.Linear(num_nodes * input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.state_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_nodes * 2),
        )

        self.param_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_edges * 2),
        )

    def forward(
        self,
        measurements: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            measurements: Dict with 'v_mag', 'p_bus', 'q_bus' each of
                shape [batch, seq_len, num_nodes].
            edge_index: Shape [2, num_edges].
            edge_attr: Optional, shape [num_edges, edge_dim].
            obs_mask: Optional, shape [batch, seq_len, num_nodes].

        Returns:
            Tuple of (states, params) where states has 'v_mag' and
            'v_ang' of shape [batch, num_nodes], and params has
            'r_line' and 'x_line' of shape [batch, num_edges].
        """
        v_mag = torch.nan_to_num(measurements["v_mag"])
        p_bus = torch.nan_to_num(measurements["p_bus"])
        q_bus = torch.nan_to_num(measurements["q_bus"])

        x = torch.stack([v_mag, p_bus, q_bus], dim=-1)
        if obs_mask is not None:
            x = x * obs_mask.unsqueeze(-1).float()

        batch_size = x.size(0)
        seq_len = x.size(1)
        x_flat = x.reshape(batch_size, seq_len, -1)
        x_proj = self.input_proj(x_flat)

        h = self.transformer(x_proj)
        h_pooled = h.mean(dim=1)

        state_out = self.state_head(h_pooled)
        state_out = state_out.reshape(batch_size, self.num_nodes, 2)
        v_mag_pred = torch.sigmoid(state_out[..., 0]) * 0.3 + 0.85
        v_ang_pred = torch.tanh(state_out[..., 1]) * 0.5
        states = {"v_mag": v_mag_pred, "v_ang": v_ang_pred}

        param_out = self.param_head(h_pooled)
        param_out = param_out.reshape(batch_size, self.num_edges, 2)
        r_line = torch.sigmoid(param_out[..., 0]) * 2.99 + 0.01
        x_line = torch.sigmoid(param_out[..., 1]) * 2.99 + 0.01
        params = {"r_line": r_line, "x_line": x_line}

        return states, params


if __name__ == "__main__":
    batch_size, seq_len = 4, 10
    num_nodes, num_edges = 15, 28

    measurements = {
        "v_mag": torch.randn(batch_size, seq_len, num_nodes),
        "p_bus": torch.randn(batch_size, seq_len, num_nodes),
        "q_bus": torch.randn(batch_size, seq_len, num_nodes),
    }
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    for name, cls in [
        ("WLSEstimator", WLSEstimator),
        ("GNNEstimator", GNNEstimator),
        ("LSTMEstimator", LSTMEstimator),
        ("TransformerEstimator", TransformerEstimator),
    ]:
        model = cls(num_nodes=num_nodes, num_edges=num_edges)
        states, params = model(measurements, edge_index)
        print("%s output:" % name)
        print("  v_mag:  %s" % str(states["v_mag"].shape))
        print("  v_ang:  %s" % str(states["v_ang"].shape))
        print("  r_line: %s" % str(params["r_line"].shape))
        print("  x_line: %s" % str(params["x_line"].shape))
        print()
