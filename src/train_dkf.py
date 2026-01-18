import torch
import torch.nn as nn
import torch.optim as optim
import pandapower.networks as pn
import os
import matplotlib.pyplot as plt
import numpy as np

from dkf import DifferentiableESKF

def train_dkf(debug=False):
    # 1. Load Data
    print("Loading data...")
    if not os.path.exists("data/robust_train.pt"):
        print("Data not found!")
        return
        
    data = torch.load("data/robust_train.pt")
    v_mag_gt = data['v_mag'] 
    v_ang_gt = data['v_ang']
    p_meas = data['p_meas']
    q_meas = data['q_meas']
    
    N_seq, T, N_bus = v_mag_gt.shape
    
    # Construct Z vector (Standard + Pseudo)
    z_standard = torch.cat([p_meas, q_meas], dim=2) 
    pseudo_val = torch.tensor([1.0, 0.0]).view(1, 1, 2).repeat(N_seq, T, 1)
    z_input = torch.cat([z_standard, pseudo_val], dim=2) 
    
    # Target State
    x_target = torch.cat([v_mag_gt, v_ang_gt], dim=2) 
    
    # 2. Initialize Model
    net_pp = pn.case33bw()
    model = DifferentiableESKF(net_pp, n_history=5)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Settings
    if debug:
        epochs = 2
        batch_size = 2
        # Use only small subset
        N_seq = 4
        z_input = z_input[:N_seq, :20, :] # Shorten sequence too
        x_target = x_target[:N_seq, :20, :]
        print("DEBUG MODE: Training on 4 sequences of length 20 for 2 epochs.")
    else:
        epochs = 20
        batch_size = 10
    
    loss_history = []
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Mini-batching
        indices = torch.randperm(N_seq)
        
        for i in range(0, N_seq, batch_size):
            idx = indices[i:i+batch_size]
            z_batch = z_input[idx] 
            x_gt_batch = x_target[idx] 
            
            optimizer.zero_grad()
            
            # Forward Pass
            x_est_seq = model(z_batch)
            
            # Loss
            est_mag = x_est_seq[..., :N_bus]
            est_ang = x_est_seq[..., N_bus:]
            gt_mag = x_gt_batch[..., :N_bus]
            gt_ang = x_gt_batch[..., N_bus:]
            
            loss_mag = nn.MSELoss()(est_mag, gt_mag)
            
            diff_ang = est_ang - gt_ang
            diff_ang = torch.remainder(diff_ang + torch.pi, 2*torch.pi) - torch.pi
            loss_ang = torch.mean(diff_ang**2)
            
            loss = loss_mag + loss_ang
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / (N_seq / batch_size)
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
    # Save Model
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), "models/dkf_weights.pth")
    print("Model saved to models/dkf_weights.pth")
    
    # Plot Loss
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.savefig("plots/training_loss.png")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    train_dkf(debug=args.debug)
