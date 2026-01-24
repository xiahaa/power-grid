import pandapower as pp
import pandapower.networks as pn
import numpy as np
import torch
import os
import copy

def get_topology_A():
    net = pn.case33bw()
    return net

def modify_topology(net):
    """
    Opens line 7-8 and closes/adds line 21-8 (8-21).
    Index of line 7-8 is 7.
    """
    # Open line 7-8
    # We find the line connecting 7 and 8
    line_7_8 = net.line[(net.line.from_bus == 7) & (net.line.to_bus == 8)]
    if line_7_8.empty:
        # try reverse
        line_7_8 = net.line[(net.line.from_bus == 8) & (net.line.to_bus == 7)]

    if not line_7_8.empty:
        idx = line_7_8.index[0]
        net.line.at[idx, 'in_service'] = False

    # Add line 8-21
    # Check if it already exists (unlikely in standard case33bw as verified, but safe check)
    line_8_21 = net.line[((net.line.from_bus == 8) & (net.line.to_bus == 21)) |
                         ((net.line.from_bus == 21) & (net.line.to_bus == 8))]

    if line_8_21.empty:
        pp.create_line_from_parameters(net, from_bus=8, to_bus=21, length_km=1.0,
                                       r_ohm_per_km=2.0, x_ohm_per_km=2.0,
                                       c_nf_per_km=0, max_i_ka=1000)
    else:
        idx = line_8_21.index[0]
        net.line.at[idx, 'in_service'] = True

    return net

def generate_samples(net, n_samples=2000):

    data_list = []

    # Base loads
    p_load_base = net.load.p_mw.values.copy()
    q_load_base = net.load.q_mvar.values.copy()

    # Identifying branches for measurements
    # We will use ALL lines in the net, assuming sensors on all branches
    # If a line is out of service, its flow is 0, but we should probably still include it in the vector structure
    # to maintain constant size, or filter.
    # However, GNNs usually handle graph structure.
    # But for "Input Z" to be a vector for DAE/transfer learning, fixed size is easier.
    # The prompt implies a fixed feature vector Z.
    # "Real-time Measurements: Active/Reactive power flow (P_ij, Q_ij) on all branches."

    # Let's run the loop
    successful_samples = 0

    # Pre-calculate number of measurements
    # Real-time: 2 * num_lines (P, Q).
    # Pseudo: 2 * num_buses (P_load, Q_load).
    # Total features per node? Or flattened?
    # For GNN, we usually have node features (P_load, Q_load) and edge features (P_flow, Q_flow).
    # The prompt says: "Map the measurement features Z (size N x F) to hidden dimension".
    # This implies Z is node-based? Or Z is the full set of measurements?
    # "Input Z" usually implies the observation vector.
    # If using GNN, we attach features to nodes and edges.
    # But later for DAE it says "Measurement vector Z (Flattened)".
    # So I will store the raw data in a structured way, then format it for GNN/DAE later.

    z_list = [] # List of dictionaries or objects
    y_list = []

    while successful_samples < n_samples:
        # Randomize loads (50% to 150%)
        scale = np.random.uniform(0.5, 1.5, size=len(p_load_base))
        net.load.p_mw = p_load_base * scale
        net.load.q_mvar = q_load_base * scale

        try:
            pp.runpp(net)
        except:
            continue

        # Ground Truth Y
        # V magnitude (pu), V angle (degrees -> radians usually preferred for ML, but let's check prompt)
        # Prompt: "Voltage Magnitude (V) and Voltage Angle (theta)"
        v_mag = net.res_bus.vm_pu.values # Shape (N_bus,)
        v_ang = np.deg2rad(net.res_bus.va_degree.values) # Shape (N_bus,)

        y_sample = np.stack([v_mag, v_ang], axis=1) # (N_bus, 2)

        # Real-time Measurements (Branch Flows)
        # We take 'p_from_mw' and 'q_from_mvar' for all lines
        # NOTE: If topology changes, the number of lines might change if we ADD a line.
        # But for Topology A, it's fixed. For Topology B, we add 1 line.
        # To make DAE work (Flattened Z), Z size must be constant.
        # This is a tricky point.
        # Usually, state estimators work on the specific graph.
        # But DAE for topology detection needs a consistent input vector to detect changes.
        # If the input vector changes size, the DAE won't work directly.
        # Maybe we assume the "Potential" lines are the set of all possible lines?
        # Or, for Topology B (transfer learning), we might just use the same "sensors" as A?
        # The prompt says: "Detect when the grid topology changes... Train on Topology A... feed Topology B".
        # This implies the input vector Z has the same dimension.
        # If I add a line in Pandapower, `net.line` grows.
        # I should probably fix the set of lines to be the UNION of A and B, or just A's lines.
        # But if I use A's lines, I won't see flow on the new line 21-8.
        # HOWEVER, the DAE is detecting the change.
        # If 7-8 opens, flow becomes 0. That's a change.
        # If 21-8 closes, flow appears where there was none?
        # Let's assume we monitor the set of lines present in Topology A.
        # If 21-8 is NOT in Topology A, we might not monitor it for the DAE trained on A.
        # BUT for the State Estimator (GSJKN), we might want to know about it.
        # The prompt says "GarL... Adapt ... to Topology B".
        # If the graph structure changes, the GNN structure changes.

        # Strategy:
        # For DAE: Use measurements from lines in Topology A.
        # For GSJKN: Use the actual graph structure (Node features + Edge features).

        # So I will save:
        # 1. Edge Index (Graph structure)
        # 2. Edge Attributes (Flow measurements)
        # 3. Node Attributes (Pseudo measurements)

        # Real-time measurements (Flows)
        # We need to capture flows for ALL lines currently in the net.
        p_flow = net.res_line.p_from_mw.values
        q_flow = net.res_line.q_from_mvar.values

        # Add 1% Uniform Noise
        # "1% Uniform Noise" -> Value * (1 + U(-0.01, 0.01)) ? Or % of range?
        # Usually % of reading.
        # "Add 1% Uniform Noise"
        noise_flow_p = np.random.uniform(-0.01, 0.01, size=len(p_flow))
        noise_flow_q = np.random.uniform(-0.01, 0.01, size=len(q_flow))

        p_flow_meas = p_flow * (1 + noise_flow_p)
        q_flow_meas = q_flow * (1 + noise_flow_q)

        # Anomalies: Mask 20% of Real-time measurements
        mask_indices = np.random.choice(len(p_flow), size=int(0.2 * len(p_flow)), replace=False)
        p_flow_meas[mask_indices] = 0.0
        q_flow_meas[mask_indices] = 0.0

        # Pseudo-Measurements (Bus Loads)
        # These are the INPUT loads to the simulation, plus noise.
        # net.load maps to buses. We need a vector for ALL buses (some might have 0 load).
        # Initialize with 0
        p_load_meas = np.zeros(len(net.bus))
        q_load_meas = np.zeros(len(net.bus))

        # Map loads to buses
        # net.load has 'bus' column
        load_buses = net.load.bus.values
        # The values we set:
        actual_p_load = net.load.p_mw.values
        actual_q_load = net.load.q_mvar.values

        # Add 50% Uniform Noise
        noise_load_p = np.random.uniform(-0.5, 0.5, size=len(actual_p_load))
        noise_load_q = np.random.uniform(-0.5, 0.5, size=len(actual_q_load))

        p_load_noisy = actual_p_load * (1 + noise_load_p)
        q_load_noisy = actual_q_load * (1 + noise_load_q)

        # Assign to full bus vector
        # Note: multiple loads can exist on one bus, usually we sum them.
        # In case33bw, usually 1 load per bus.
        np.add.at(p_load_meas, load_buses, p_load_noisy)
        np.add.at(q_load_meas, load_buses, q_load_noisy)

        # Store Data
        # We need edge_index for PyG
        # From pandapower net.line
        # from_bus, to_bus
        edge_start = net.line.from_bus.values
        edge_end = net.line.to_bus.values
        edge_index = np.vstack([edge_start, edge_end])

        # We store everything in a dict
        sample = {
            'x': np.stack([p_load_meas, q_load_meas], axis=1), # Node features (Pseudo-meas)
            'edge_index': edge_index,
            'edge_attr': np.stack([p_flow_meas, q_flow_meas], axis=1), # Edge features (Real-time meas)
            'y': y_sample, # Ground truth (V, theta)
            'line_status': net.line.in_service.values.astype(float) # Track topology if needed
        }

        data_list.append(sample)
        successful_samples += 1

        if successful_samples % 100 == 0:
            print(f"Generated {successful_samples}/{n_samples}")

    return data_list

def save_dataset(data_list, filename):
    # Convert list of dicts to list of PyG Data objects?
    # Or just save the list of dicts for now and process in dataset class.
    # User said: "Generate a dataset... data_generator.py"
    # "Required packages... torch_geometric".
    # It's better to save as a list of Data objects or a collated object.
    # Let's save as list of Data objects.

    from torch_geometric.data import Data

    pyg_list = []
    for s in data_list:
        # Node features: [N, 2]
        x = torch.tensor(s['x'], dtype=torch.float)
        # Edge index: [2, E]
        edge_index = torch.tensor(s['edge_index'], dtype=torch.long)
        # Edge attr: [E, 2]
        edge_attr = torch.tensor(s['edge_attr'], dtype=torch.float)
        # Y: [N, 2]
        y = torch.tensor(s['y'], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        pyg_list.append(data)

    torch.save(pyg_list, filename)
    print(f"Saved {filename}")

def main():
    print("Generating Topology A (Standard)...")
    net_A = get_topology_A()
    data_A = generate_samples(net_A, n_samples=2000)
    save_dataset(data_A, 'dataset_topology_A.pt')

    print("Generating Topology B (Modified)...")
    # Reset net
    net_B = get_topology_A()
    net_B = modify_topology(net_B)
    data_B = generate_samples(net_B, n_samples=50)
    save_dataset(data_B, 'dataset_topology_B.pt')

if __name__ == "__main__":
    main()
