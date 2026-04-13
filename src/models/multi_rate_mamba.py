"""
Multi-Rate Mamba Fusion Network for Distribution System State Estimation

Core Innovation: Dual-stream Mamba architecture that processes SCADA and PMU
measurements at their native sampling rates, then fuses them via cross-attention.

Architecture:
    SCADA Stream (slow, 0.25 Hz) --> Slow Mamba -->    \
                                                          --> Cross-Attention --> State/Parameter Heads
    PMU Stream (fast, 60 Hz)   --> Fast Mamba -->      /

Key advantage over single-rate approaches:
- PMU data preserves high-frequency dynamics (60 Hz phasors)
- SCADA data provides wide spatial coverage (60-80% of buses)
- Mamba's O(n) complexity handles 240x rate difference efficiently
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class TemporalAlignmentLayer(nn.Module):
    """Align multi-rate sequences to a common temporal resolution."""

    def __init__(self, slow_dim: int, fast_dim: int, target_len: int = 10):
        super().__init__()
        self.target_len = target_len
        self.slow_proj = nn.Linear(slow_dim, slow_dim)
        self.fast_proj = nn.Linear(fast_dim, fast_dim)

    def forward(
        self, slow_seq: torch.Tensor, fast_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        slow_aligned = self.slow_proj(slow_seq)
        if slow_seq.size(1) < self.target_len:
            slow_aligned = F.interpolate(
                slow_aligned.transpose(1, 2),
                size=self.target_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        elif slow_seq.size(1) > self.target_len:
            slow_aligned = F.adaptive_avg_pool1d(
                slow_aligned.transpose(1, 2), self.target_len
            ).transpose(1, 2)

        fast_aligned = self.fast_proj(fast_seq)
        if fast_seq.size(1) > self.target_len:
            fast_aligned = F.adaptive_avg_pool1d(
                fast_aligned.transpose(1, 2), self.target_len
            ).transpose(1, 2)
        elif fast_seq.size(1) < self.target_len:
            fast_aligned = F.interpolate(
                fast_aligned.transpose(1, 2),
                size=self.target_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        return slow_aligned, fast_aligned


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between slow (SCADA) and fast (PMU) streams."""

    def __init__(
        self,
        slow_dim: int,
        fast_dim: int,
        fusion_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.slow_proj = nn.Linear(slow_dim, fusion_dim)
        self.fast_proj = nn.Linear(fast_dim, fusion_dim)

        self.cross_attn_fast_to_slow = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_slow_to_fast = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm_fast = nn.LayerNorm(fusion_dim)
        self.norm_slow = nn.LayerNorm(fusion_dim)

        self.gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Sigmoid())
        self.output_proj = nn.Linear(fusion_dim * 2, fusion_dim)

    def forward(
        self,
        slow_features: torch.Tensor,
        fast_features: torch.Tensor,
        slow_mask: Optional[torch.Tensor] = None,
        fast_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_slow = self.slow_proj(slow_features)
        h_fast = self.fast_proj(fast_features)

        attn_fast, _ = self.cross_attn_fast_to_slow(h_fast, h_slow, h_slow)
        attn_fast = self.norm_fast(h_fast + attn_fast)

        attn_slow, _ = self.cross_attn_slow_to_fast(h_slow, h_fast, h_fast)
        attn_slow = self.norm_slow(h_slow + attn_slow)

        h_fast_pooled = attn_fast.mean(dim=1)
        h_slow_pooled = attn_slow.mean(dim=1)

        gate_input = torch.cat([h_fast_pooled, h_slow_pooled], dim=-1)
        gate_weights = self.gate(gate_input)

        h_fast_gated = h_fast_pooled * gate_weights
        h_slow_gated = h_slow_pooled * (1 - gate_weights)

        fused = self.output_proj(torch.cat([h_fast_gated, h_slow_gated], dim=-1))
        return fused


class MultiRateMambaFusion(nn.Module):
    """
    Multi-Rate Mamba Fusion for DSSE.

    Processes SCADA (slow) and PMU (fast) measurements in parallel
    Mamba streams and fuses via cross-attention.

    Args:
        num_nodes: Number of buses
        num_edges: Number of lines (x2 for bidirectional)
        input_dim: Input feature dimension per node
        spatial_config: GAT spatial encoder config
        pmu_stream_config: Fast Mamba stream config
        scada_stream_config: Slow Mamba stream config
        fusion_config: Cross-attention fusion config
        state_head_config: State estimation head config
        parameter_head_config: Parameter estimation head config
    """

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        input_dim: int = 4,
        spatial_hidden_dim: int = 64,
        spatial_num_heads: int = 4,
        spatial_num_layers: int = 2,
        pmu_d_model: int = 128,
        pmu_d_state: int = 32,
        pmu_num_layers: int = 3,
        scada_d_model: int = 64,
        scada_d_state: int = 8,
        scada_num_layers: int = 2,
        fusion_dim: int = 128,
        fusion_num_heads: int = 4,
        state_hidden_dims: list = None,
        parameter_hidden_dims: list = None,
        feeder_map: Optional[Dict[int, list]] = None,
        feeder_emb_dim: int = 0,
        feeder_refine_hidden_dim: int = 32,
        measurement_refine_dim: int = 0,
        feeder_target_ids: Optional[list] = None,
        feeder_refine_scale: float = 1.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.feeder_emb_dim = feeder_emb_dim if feeder_map else 0
        self.measurement_refine_dim = measurement_refine_dim
        self.feeder_refine_scale = feeder_refine_scale

        from torch_geometric.nn import GATConv

        spatial_out_dim = spatial_hidden_dim * spatial_num_heads

        self.input_dim = input_dim
        self.input_proj = nn.Linear(input_dim, spatial_hidden_dim)

        self.spatial_convs = nn.ModuleList()
        for i in range(spatial_num_layers):
            in_dim = spatial_hidden_dim if i == 0 else spatial_out_dim
            self.spatial_convs.append(
                GATConv(
                    in_dim,
                    spatial_hidden_dim,
                    heads=spatial_num_heads,
                    dropout=0.1,
                    concat=True,
                )
            )
        self.spatial_dropout = nn.Dropout(0.1)

        if MAMBA_AVAILABLE:
            self.pmu_temporal = nn.ModuleList(
                [
                    Mamba(
                        d_model=spatial_out_dim, d_state=pmu_d_state, d_conv=4, expand=2
                    )
                    for _ in range(pmu_num_layers)
                ]
            )
            self.scada_temporal = nn.ModuleList(
                [
                    Mamba(
                        d_model=spatial_out_dim,
                        d_state=scada_d_state,
                        d_conv=4,
                        expand=2,
                    )
                    for _ in range(scada_num_layers)
                ]
            )
        else:
            self.pmu_temporal = nn.ModuleList(
                [
                    nn.LSTM(spatial_out_dim, spatial_out_dim, batch_first=True)
                    for _ in range(pmu_num_layers)
                ]
            )
            self.scada_temporal = nn.ModuleList(
                [
                    nn.LSTM(spatial_out_dim, spatial_out_dim, batch_first=True)
                    for _ in range(scada_num_layers)
                ]
            )

        self.pmu_norm = nn.LayerNorm(spatial_out_dim)
        self.scada_norm = nn.LayerNorm(spatial_out_dim)

        self.fusion = CrossAttentionFusion(
            slow_dim=spatial_out_dim,
            fast_dim=spatial_out_dim,
            fusion_dim=fusion_dim,
            num_heads=fusion_num_heads,
        )

        state_hidden_dims = state_hidden_dims or [128, 64]
        param_hidden_dims = parameter_hidden_dims or [128, 64]

        self.state_head = self._build_mlp(fusion_dim, state_hidden_dims, 2)
        self.register_buffer("feeder_ids", None)
        self.register_buffer("feeder_target_mask", None)
        self.feeder_embedding = None
        self.feeder_vmag_refine = None
        if feeder_map and self.feeder_emb_dim > 0:
            feeder_ids = torch.zeros(num_nodes, dtype=torch.long)
            feeder_target_mask = torch.ones(num_nodes, dtype=torch.float32)
            if feeder_target_ids is not None:
                feeder_target_mask.zero_()
            for feeder_id, buses in feeder_map.items():
                for bus in buses:
                    idx = bus - 1
                    if 0 <= idx < num_nodes:
                        feeder_ids[idx] = int(feeder_id)
                        if feeder_target_ids is not None and feeder_id in feeder_target_ids:
                            feeder_target_mask[idx] = 1.0
            self.feeder_ids = feeder_ids
            self.feeder_target_mask = feeder_target_mask
            num_feeders = max(feeder_map.keys()) + 1 if feeder_map else 1
            self.feeder_embedding = nn.Embedding(num_feeders, self.feeder_emb_dim)
            self.feeder_vmag_refine = self._build_mlp(
                fusion_dim + self.feeder_emb_dim + self.measurement_refine_dim,
                [feeder_refine_hidden_dim],
                1,
            )
        self.parameter_head_edge_pool = nn.Linear(fusion_dim * 2, fusion_dim)
        self.parameter_head = self._build_mlp(fusion_dim, param_hidden_dims, 2)

    def _build_mlp(self, in_dim, hidden_dims, out_dim):
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.LayerNorm(h)])
            prev = h
        final = nn.Linear(prev, out_dim)
        nn.init.xavier_uniform_(final.weight, gain=0.01)
        nn.init.zeros_(final.bias)
        layers.append(final)
        return nn.Sequential(*layers)

    def _spatial_encode(self, x, edge_index, edge_attr=None):
        batch_size, num_nodes, _ = x.shape
        h_list = []
        for b in range(batch_size):
            h = x[b]
            for conv in self.spatial_convs:
                h = conv(h, edge_index, edge_attr=edge_attr)
                h = F.elu(h)
                h = self.spatial_dropout(h)
            h_list.append(h)
        return torch.stack(h_list, dim=0)

    def _temporal_encode_pmu(self, x):
        h = x
        for layer in self.pmu_temporal:
            if hasattr(layer, "A_log"):
                h = h + layer(self.pmu_norm(h))
            else:
                out, _ = layer(h)
                h = h + out
        return h

    def _temporal_encode_scada(self, x):
        h = x
        for layer in self.scada_temporal:
            if hasattr(layer, "A_log"):
                h = h + layer(self.scada_norm(h))
            else:
                out, _ = layer(h)
                h = h + out
        return h

    def _build_measurement_refine_features(
        self,
        obs_mask: Optional[torch.Tensor],
        scada_obs_mask: Optional[torch.Tensor],
        pmu_obs_mask: Optional[torch.Tensor],
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.measurement_refine_dim <= 0:
            return None

        feature_list = []

        def _reduce(mask: Optional[torch.Tensor]) -> torch.Tensor:
            if mask is None:
                return torch.zeros(batch_size, num_nodes, device=device)
            if mask.dim() == 3:
                return mask.float().mean(dim=1).to(device)
            return mask.float().to(device)

        if self.measurement_refine_dim >= 1:
            feature_list.append(_reduce(pmu_obs_mask))
        if self.measurement_refine_dim >= 2:
            feature_list.append(_reduce(scada_obs_mask))
        if self.measurement_refine_dim >= 3:
            feature_list.append(_reduce(obs_mask))

        while len(feature_list) < self.measurement_refine_dim:
            feature_list.append(torch.zeros(batch_size, num_nodes, device=device))

        return torch.stack(feature_list[: self.measurement_refine_dim], dim=-1)

    def forward(
        self,
        measurements: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
        scada_obs_mask: Optional[torch.Tensor] = None,
        pmu_obs_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch_size, seq_len, num_nodes = measurements["v_mag"].shape

        v_ang = measurements.get("v_ang")
        if v_ang is None:
            v_ang = torch.zeros_like(measurements["v_mag"])

        feature_tensors = [
            measurements["v_mag"],
            measurements["p_bus"],
            measurements["q_bus"],
            v_ang,
        ]

        if self.input_dim < len(feature_tensors):
            x = torch.stack(feature_tensors[: self.input_dim], dim=-1)
        elif self.input_dim > len(feature_tensors):
            padding = [
                torch.zeros_like(measurements["v_mag"])
                for _ in range(self.input_dim - len(feature_tensors))
            ]
            x = torch.stack(feature_tensors + padding, dim=-1)
        else:
            x = torch.stack(feature_tensors, dim=-1)

        if obs_mask is not None:
            x = x * obs_mask.unsqueeze(-1)

        x = self.input_proj(x)

        spatial_features = []
        for t in range(seq_len):
            h_spatial = self._spatial_encode(x[:, t], edge_index, edge_attr)
            spatial_features.append(h_spatial)
        spatial_features = torch.stack(spatial_features, dim=1)

        pmu_flat = spatial_features.reshape(batch_size * num_nodes, seq_len, -1)
        pmu_encoded = self._temporal_encode_pmu(pmu_flat)
        pmu_features = pmu_encoded.reshape(batch_size, num_nodes, seq_len, -1)

        scada_stride = max(1, seq_len // 5)
        scada_indices = list(range(0, seq_len, scada_stride))
        scada_input = spatial_features[:, scada_indices, :, :]
        _, scada_seq_len, _, feat_dim = scada_input.shape
        scada_flat = scada_input.reshape(
            batch_size * num_nodes, scada_seq_len, feat_dim
        )
        scada_encoded = self._temporal_encode_scada(scada_flat)
        scada_features = scada_encoded.reshape(
            batch_size, num_nodes, scada_seq_len, feat_dim
        )

        all_fused = []
        for n in range(num_nodes):
            pmu_node = pmu_features[:, n, :, :]
            scada_node = scada_features[:, n, :, :]
            fused_node = self.fusion(scada_node, pmu_node)
            all_fused.append(fused_node)

        fused = torch.stack(all_fused, dim=1)

        state_out = self.state_head(fused)
        v_mag_logits = state_out[..., 0]
        if self.feeder_embedding is not None and self.feeder_vmag_refine is not None:
            feeder_emb = self.feeder_embedding(self.feeder_ids)
            feeder_emb = feeder_emb.unsqueeze(0).expand(batch_size, -1, -1)
            refine_parts = [fused, feeder_emb]
            measurement_features = self._build_measurement_refine_features(
                obs_mask=obs_mask,
                scada_obs_mask=scada_obs_mask,
                pmu_obs_mask=pmu_obs_mask,
                batch_size=batch_size,
                num_nodes=num_nodes,
                device=fused.device,
            )
            if measurement_features is not None:
                refine_parts.append(measurement_features)

            refine_input = torch.cat(refine_parts, dim=-1)
            refine_delta = self.feeder_vmag_refine(refine_input).squeeze(-1)
            if self.feeder_target_mask is not None:
                refine_delta = refine_delta * self.feeder_target_mask.unsqueeze(0)
            v_mag_logits = v_mag_logits + self.feeder_refine_scale * refine_delta

        v_mag = torch.sigmoid(v_mag_logits) * 0.3 + 0.85
        v_ang = torch.tanh(state_out[..., 1]) * 0.5

        states = {"v_mag": v_mag, "v_ang": v_ang}

        from_nodes = edge_index[0]
        to_nodes = edge_index[1]

        unique_edges = []
        seen = set()
        for i in range(from_nodes.shape[0]):
            a, b = from_nodes[i].item(), to_nodes[i].item()
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                unique_edges.append((a, b))
        if not unique_edges:
            unique_edges = list(zip(from_nodes.tolist(), to_nodes.tolist()))

        ue_from = torch.tensor([e[0] for e in unique_edges], device=from_nodes.device)
        ue_to = torch.tensor([e[1] for e in unique_edges], device=from_nodes.device)
        num_unique = len(unique_edges)

        edge_feat_list = []
        for b in range(batch_size):
            from_feat = fused[b, ue_from]
            to_feat = fused[b, ue_to]
            edge_feat = self.parameter_head_edge_pool(
                torch.cat([from_feat, to_feat], dim=-1)
            )
            edge_feat_list.append(edge_feat)
        edge_feat = torch.stack(edge_feat_list, dim=0)

        param_out = self.parameter_head(edge_feat)
        log_r_line = torch.clamp(param_out[..., 0], min=-12.0, max=16.0)
        log_x_line = torch.clamp(param_out[..., 1], min=-12.0, max=16.0)
        r_line = torch.exp(log_r_line)
        x_line = torch.exp(log_x_line)

        parameters = {"r_line": r_line, "x_line": x_line}

        return states, parameters


if __name__ == "__main__":
    model = MultiRateMambaFusion(
        num_nodes=123,
        num_edges=240,
        input_dim=4,
        spatial_hidden_dim=64,
        pmu_d_model=128,
        scada_d_model=64,
    )

    batch_size, seq_len = 4, 10
    measurements = {
        "v_mag": torch.randn(batch_size, seq_len, 123),
        "p_bus": torch.randn(batch_size, seq_len, 123),
        "q_bus": torch.randn(batch_size, seq_len, 123),
        "v_ang": torch.randn(batch_size, seq_len, 123),
    }
    edge_index = torch.randint(0, 123, (2, 240))

    states, parameters = model(measurements, edge_index)
    print("MultiRateMambaFusion output:")
    print(f"  V_mag: {states['v_mag'].shape}")
    print(f"  V_ang: {states['v_ang'].shape}")
    print(f"  R_line: {parameters['r_line'].shape}")
    print(f"  X_line: {parameters['x_line'].shape}")
