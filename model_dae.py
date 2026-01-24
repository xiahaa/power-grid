import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class DAE(nn.Module):
    def __init__(self, input_dim):
        super(DAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

def get_flattened_input(data, fixed_num_edges=37):
    """
    Extracts and flattens Z from a PyG Data object.
    Z = [Edge_Flows, Node_Loads]
    Truncates/Pads edge flows to fixed_num_edges.
    """
    # Edge Attr: [E, 2]
    e_attr = data.edge_attr
    if e_attr.size(0) > fixed_num_edges:
        e_attr = e_attr[:fixed_num_edges, :]
    elif e_attr.size(0) < fixed_num_edges:
        # Pad with zeros (though typically not needed if training on A)
        padding = torch.zeros(fixed_num_edges - e_attr.size(0), 2, device=e_attr.device)
        e_attr = torch.cat([e_attr, padding], dim=0)

    e_flat = e_attr.view(-1)

    # Node X: [N, 2]
    # N is usually 33.
    x_flat = data.x.view(-1)

    z = torch.cat([e_flat, x_flat])
    return z

def train_and_evaluate_dae():
    # Load Data
    print("Loading data...")
    data_A = torch.load('dataset_topology_A.pt', weights_only=False)
    data_B = torch.load('dataset_topology_B.pt', weights_only=False)

    # Determine Input Dim
    sample_z = get_flattened_input(data_A[0])
    input_dim = sample_z.size(0)
    print(f"DAE Input Dimension: {input_dim}")

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DAE(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()

    # Train on A (Use 80% split just to be consistent, or full A?
    # Usually anomaly detection trains on "normal" data. So all A is fine, or Train A.)
    # Let's use 100% of A for training to learn "Normal" well, or split to avoid overfitting.
    # The prompt says "Train on Topology A".
    train_loader = DataLoader(data_A, batch_size=32, shuffle=True)

    print("Training DAE on Topology A...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)

            # Create batch of inputs
            # get_flattened_input works on single sample.
            # DataLoader batches graphs.
            # We can process the batch. data.edge_attr is [Batch*E_per_graph, 2].
            # But graph sizes vary slightly due to edge truncation needed?
            # Actually, Topology A always has 37 edges.
            # So we can just reshape.

            # Efficient batch processing:
            # We know A has 33 nodes, 37 edges.
            batch_size = data.num_graphs
            # Edge attr: [B * 37, 2] -> [B, 37*2]
            e_flat = data.edge_attr.view(batch_size, -1)
            # Node x: [B * 33, 2] -> [B, 33*2]
            x_flat = data.x.view(batch_size, -1)

            z = torch.cat([e_flat, x_flat], dim=1)

            optimizer.zero_grad()
            z_recon = model(z)
            loss = criterion(z_recon, z)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size

        avg_loss = total_loss / len(data_A)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # Test on Sequence: 50 samples of A (Test set) + 50 samples of B
    print("Testing on mixed sequence...")
    test_data = data_A[-50:] + data_B # 50 A + 50 B

    reconstruction_errors = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_data):
            data = data.to(device)
            z = get_flattened_input(data).unsqueeze(0) # [1, Input_Dim]
            z_recon = model(z)
            loss = criterion(z_recon, z) # Scalar L1
            reconstruction_errors.append(loss.item())

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(reconstruction_errors, label='Reconstruction Error')
    plt.axvline(x=50, color='r', linestyle='--', label='Topology Change')
    plt.xlabel('Sample Index')
    plt.ylabel('L1 Reconstruction Error')
    plt.title('DAE Anomaly Detection')
    plt.legend()
    plt.savefig('dae_anomaly_detection.png')
    print("Saved plot to dae_anomaly_detection.png")

if __name__ == "__main__":
    train_and_evaluate_dae()
