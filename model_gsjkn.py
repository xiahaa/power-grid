import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch

class GSJKN(nn.Module):
    def __init__(self, node_in_dim=2, edge_in_dim=2, hidden_dim=16, num_layers=3, rnn_dim=32, out_dim=2):
        super(GSJKN, self).__init__()

        self.num_layers = num_layers

        # Input Embedding
        self.node_emb = nn.Linear(node_in_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_in_dim, hidden_dim)

        # GAT Layers
        self.gats = nn.ModuleList()
        for _ in range(num_layers):
            # GATConv with edge attributes
            # heads=1 for simplicity unless specified.
            self.gats.append(GATConv(hidden_dim, hidden_dim, heads=1, edge_dim=hidden_dim))

        # Global Scanning (Bi-RNN)
        # Input to RNN is concatenation of all GAT layers: num_layers * hidden_dim
        jk_dim = num_layers * hidden_dim
        self.rnn = nn.GRU(input_size=jk_dim, hidden_size=rnn_dim, batch_first=True, bidirectional=True)

        # Estimator Head
        # Input: 2 * rnn_dim (Bi-Directional)
        # Output: out_dim (V, theta)
        self.head = nn.Sequential(
            nn.Linear(2 * rnn_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 1. Input Mapping
        h = self.node_emb(x)
        e = self.edge_emb(edge_attr)

        # 2. Graph Jump Layer
        layer_outputs = []
        for gat in self.gats:
            h = gat(h, edge_index, edge_attr=e)
            h = F.leaky_relu(h)
            layer_outputs.append(h)

        # Concatenate JK
        h_jk = torch.cat(layer_outputs, dim=-1) # [Total_Nodes, num_layers * hidden_dim]

        # 3. Global Scanning
        # Reshape to [Batch, Num_Nodes, Features]
        # to_dense_batch handles variable number of nodes if necessary, padding with 0
        h_dense, mask = to_dense_batch(h_jk, batch) # [B, N_max, F]

        # RNN
        # h_dense is [Batch, Seq_Len, F]
        # We assume the node order in 'to_dense_batch' respects the node indexing (0..32).
        # PyG batching preserves node order within each graph.
        rnn_out, _ = self.rnn(h_dense) # [B, N, 2*rnn_dim]

        # 4. Estimator Head
        # Apply to each node
        out = self.head(rnn_out) # [B, N, 2]

        # We need to flatten back to [Total_Nodes, 2] to match 'y' format in loss
        # Use mask to select valid nodes
        out_flat = out[mask]

        return out_flat
