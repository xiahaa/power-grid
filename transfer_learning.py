import torch
import numpy as np
from model_gsjkn import GSJKN
from model_dae import get_flattened_input # We can reuse or rewrite if we want 38 edges
from torch_geometric.loader import DataLoader
import scipy.linalg

def get_flat_vectors(dataset, model, device):
    """
    Returns Z (input), Y_pseudo (prediction), Y_true (label) as flat numpy arrays.
    Z: [N_samples, Dim_Z]
    Y_pseudo: [N_samples, Dim_Y]
    Y_true: [N_samples, Dim_Y]
    """
    Z_list = []
    Y_pseudo_list = []
    Y_true_list = []

    model.eval()
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)

            # Predict
            # GSJKN output is [N_nodes * 2] (flattened in forward)
            out = model(data)

            Y_pseudo_list.append(out.cpu().numpy().flatten())
            Y_true_list.append(data.y.cpu().numpy().flatten())

            # Z Vector
            # We flatten all features in the data object
            # Node X: [33, 2] -> 66
            # Edge Attr: [38, 2] -> 76
            x_flat = data.x.view(-1)
            e_flat = data.edge_attr.view(-1)
            z_vec = torch.cat([x_flat, e_flat]).cpu().numpy()
            Z_list.append(z_vec)

    return np.array(Z_list), np.array(Y_pseudo_list), np.array(Y_true_list)

def kernel_function(Z, Yp, alpha=1.0, beta=1.0):
    """
    Computes Composite Kernel Matrix.
    K = alpha * (Z @ Z.T) + beta * (Yp @ Yp.T)
    """
    K_z = Z @ Z.T
    K_yp = Yp @ Yp.T
    return alpha * K_z + beta * K_yp

def solve_gp(K_train, R_train, sigma_noise=1e-2):
    """
    Solves (K + sigma^2 I)^-1 R
    """
    N = K_train.shape[0]
    L = np.linalg.cholesky(K_train + sigma_noise**2 * np.eye(N))
    # Solve L * y = R
    # Solve L.T * x = y
    # alpha = K^-1 R
    alpha = scipy.linalg.cho_solve((L, True), R_train)
    return alpha

def transfer_learning():
    print("Loading Topology B data...")
    dataset_B = torch.load('dataset_topology_B.pt', weights_only=False)

    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GSJKN(node_in_dim=2, edge_in_dim=2, hidden_dim=16, num_layers=3, rnn_dim=32, out_dim=2).to(device)
    model.load_state_dict(torch.load('gsjkn_model.pth', weights_only=False))

    # Get Vectors
    print("Computing predictions on B...")
    Z, Y_pseudo, Y_true = get_flat_vectors(dataset_B, model, device)

    # Split
    n_train = 20
    Z_train, Z_test = Z[:n_train], Z[n_train:]
    Yp_train, Yp_test = Y_pseudo[:n_train], Y_pseudo[n_train:]
    Yt_train, Yt_test = Y_true[:n_train], Y_true[n_train:]

    # Residuals
    R_train = Yt_train - Yp_train

    # Normalize features?
    # Linear kernel is sensitive to scale.
    # Z contains flows (~0-10 MW) and Loads (~0-5 MW).
    # Yp contains Voltage (~1.0 pu) and Angle (~0 rad).
    # It's better to normalize Z and Yp for the kernel.
    Z_mean, Z_std = Z_train.mean(0), Z_train.std(0) + 1e-6
    Yp_mean, Yp_std = Yp_train.mean(0), Yp_train.std(0) + 1e-6

    Z_train_norm = (Z_train - Z_mean) / Z_std
    Z_test_norm = (Z_test - Z_mean) / Z_std

    Yp_train_norm = (Yp_train - Yp_mean) / Yp_std
    Yp_test_norm = (Yp_test - Yp_mean) / Yp_std

    # Train GP
    # We simply compute alpha = K_inv * R
    print("Training GP (computing kernel inverse)...")
    K_train = kernel_function(Z_train_norm, Yp_train_norm)

    # We have multiple outputs (66 dimensions).
    # We solve for all of them using the same K.
    gp_alphas = solve_gp(K_train, R_train, sigma_noise=0.1)

    # Predict
    # K_star = kernel(Test, Train)
    # Z_test @ Z_train.T ...
    K_star_z = Z_test_norm @ Z_train_norm.T
    K_star_yp = Yp_test_norm @ Yp_train_norm.T
    K_star = 1.0 * K_star_z + 1.0 * K_star_yp

    R_pred = K_star @ gp_alphas

    Y_final = Yp_test + R_pred

    # Evaluation
    mse_old = np.mean((Yt_test - Yp_test)**2)
    mse_new = np.mean((Yt_test - Y_final)**2)

    print(f"Results on Topology B (30 test samples):")
    print(f"Original GSJKN MSE: {mse_old:.6f}")
    print(f"GARL Corrected MSE: {mse_new:.6f}")
    print(f"Improvement: {(mse_old - mse_new)/mse_old * 100:.2f}%")

if __name__ == "__main__":
    transfer_learning()
