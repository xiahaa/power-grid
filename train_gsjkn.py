import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model_gsjkn import GSJKN
import numpy as np
import random

def train_gsjkn():
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Load Data
    print("Loading data...")
    dataset = torch.load('dataset_topology_A.pt', weights_only=False)

    # Split
    # Shuffle first
    random.shuffle(dataset)
    split_idx = int(0.8 * len(dataset))
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GSJKN(node_in_dim=2, edge_in_dim=2, hidden_dim=16, num_layers=3, rnn_dim=32, out_dim=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Starting training...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        avg_loss = total_loss / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)
                val_loss += loss.item() * data.num_graphs
        avg_val_loss = val_loss / len(test_dataset)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    # Save model
    torch.save(model.state_dict(), 'gsjkn_model.pth')
    print("Model saved to gsjkn_model.pth")

    return avg_val_loss

if __name__ == "__main__":
    train_gsjkn()
