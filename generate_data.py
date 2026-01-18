import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import os

def create_base_net():
    # Load standard IEEE 33 bus system
    net = pn.case33bw()
    return net

def get_load_scale(step_idx, steps_per_day=96, base_scale=1.0):
    """
    Generates load scale for a specific time step.
    """
    hour = (step_idx % steps_per_day) * (24.0 / steps_per_day)
    t = hour
    # Profile: Double peak
    profile = 0.5 + 0.5 * np.sin(np.pi * (t - 6) / 12)**2 + 0.3 * np.exp(-(t - 19)**2 / 4)
    # Add noise
    noise = np.random.normal(0, 0.05)
    return (profile + noise) * base_scale

def apply_parameter_drift(net, step, total_steps):
    """
    Simulate parameter drift (e.g., aging lines).
    """
    # We modify specific lines: 2 and 5
    drift_factor = 1.0 + (0.2 * step / total_steps) # 1.0 -> 1.2

    target_lines = [2, 5]

    # The caller is responsible for resetting the net values to original before calling this.
    for line_idx in target_lines:
        net.line.at[line_idx, 'r_ohm_per_km'] *= drift_factor
        net.line.at[line_idx, 'x_ohm_per_km'] *= drift_factor

    return net

def generate_dataset(n_days=10, steps_per_day=96):
    net = create_base_net()

    # Store containers
    data_X = [] # State Ground Truth (Voltage Magnitude, Angle)
    data_Y = [] # Parameter Ground Truth (Resistance, Reactance)
    data_Z = [] # Measurements (P, Q, V_mag)

    total_steps = n_days * steps_per_day

    # Backup original parameters
    original_r = net.line['r_ohm_per_km'].copy()
    original_x = net.line['x_ohm_per_km'].copy()

    # Backup original loads
    original_p_load = net.load['p_mw'].copy()
    original_q_load = net.load['q_mvar'].copy()

    print(f"Generating data for {n_days} days ({total_steps} steps)...")

    for i in tqdm(range(total_steps)):
        # 1. Update Load
        load_scale = get_load_scale(i, steps_per_day)
        net.load['p_mw'] = original_p_load * load_scale
        net.load['q_mvar'] = original_q_load * load_scale

        # 2. Update Line Parameters (Drift)
        # Reset first
        net.line['r_ohm_per_km'] = original_r.values
        net.line['x_ohm_per_km'] = original_x.values

        apply_parameter_drift(net, i, total_steps)

        # 3. Run Power Flow
        try:
            pp.runpp(net)
        except pp.LoadflowNotConverged:
            print(f"Step {i}: Power flow did not converge. Skipping.")
            continue
        except Exception as e:
            print(f"Step {i}: Error: {e}")
            continue

        # 4. Extract Ground Truth
        # State: V magnitude (pu), V angle (degree)
        v_mag = net.res_bus['vm_pu'].values
        v_ang = net.res_bus['va_degree'].values

        # Parameters: R, X
        line_r = net.line['r_ohm_per_km'].values
        line_x = net.line['x_ohm_per_km'].values

        # 5. Generate Measurements (with noise)
        meas_p = net.res_bus['p_mw'].values + np.random.normal(0, 0.01, len(net.bus))
        meas_q = net.res_bus['q_mvar'].values + np.random.normal(0, 0.01, len(net.bus))
        meas_v = v_mag + np.random.normal(0, 0.005, len(net.bus))

        # Append to lists
        data_X.append(np.stack([v_mag, v_ang], axis=1)) # [Nodes, 2]
        data_Y.append(np.stack([line_r, line_x], axis=1)) # [Lines, 2]
        data_Z.append(np.stack([meas_p, meas_q, meas_v], axis=1)) # [Nodes, 3]

    return np.array(data_X), np.array(data_Y), np.array(data_Z), net.line.from_bus.values, net.line.to_bus.values

def save_to_pt(X_gt, Params_gt, Z_meas, edge_from, edge_to, filename="grid_data.pt"):
    data_list = []
    edge_index = torch.tensor([edge_from, edge_to], dtype=torch.long)

    print(f"Converting to PyG datasets and saving to {filename}...")

    for t in tqdm(range(len(X_gt))):
        # Input features: [P, Q, V_meas]
        x = torch.tensor(Z_meas[t], dtype=torch.float)

        # Labels:
        # State Label: [V_true, Theta_true]
        y_state = torch.tensor(X_gt[t], dtype=torch.float)
        # Parameter Label: [R_true, X_true]
        y_param = torch.tensor(Params_gt[t], dtype=torch.float)

        # Masking
        # Mask 70% of node voltage measurements (set to 0)
        # We only mask the V_meas (column index 2)
        mask = torch.rand(x.size(0)) > 0.3
        x[mask, 2] = 0.0

        data = Data(x=x, edge_index=edge_index, y_state=y_state, y_param=y_param)
        data_list.append(data)

    torch.save(data_list, filename)
    print(f"Saved {len(data_list)} snapshots to {filename}")

if __name__ == "__main__":
    # Generate 5 days of data (5 * 96 = 480 steps) for testing
    n_days = 5
    X_gt, Params_gt, Z_meas, edge_from, edge_to = generate_dataset(n_days=n_days)

    if len(X_gt) > 0:
        save_to_pt(X_gt, Params_gt, Z_meas, edge_from, edge_to, "grid_data.pt")
    else:
        print("No data generated.")
