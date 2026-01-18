import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import torch
import os

def generate_data(output_dir="data", n_steps=100):
    """
    Generates time-series data for the IEEE 33-bus system.
    Simulates load variations and solves power flow to get Ground Truth (V, theta) and Measurements (P, Q).
    """
    print("Loading IEEE 33-bus system...")
    net = pn.case33bw()
    
    # Store results
    v_mag_list = []
    v_ang_list = []
    p_meas_list = []
    q_meas_list = []
    
    # Original loads
    p_load_0 = net.load.p_mw.values.copy()
    q_load_0 = net.load.q_mvar.values.copy()
    
    print(f"Simulating {n_steps} time steps...")
    
    for t in range(n_steps):
        # Perturb loads: Random walk + sinusoidal daily pattern
        # Simple noise model: nominal * (1 + random_noise)
        noise_scale = 0.1
        noise = np.random.normal(0, noise_scale, size=len(p_load_0))
        
        # Add a time-varying component (e.g., daily cycle)
        time_factor = 1.0 + 0.2 * np.sin(2 * np.pi * t / 24.0)
        
        net.load.p_mw = p_load_0 * time_factor * (1 + noise)
        net.load.q_mvar = q_load_0 * time_factor * (1 + noise)
        
        try:
            pp.runpp(net)
            
            # Ground Truth State: Voltage Magnitude and Angle (degrees -> radians)
            v_mag = net.res_bus.vm_pu.values
            v_ang = np.deg2rad(net.res_bus.va_degree.values)
            
            # Measurements: Calculated P and Q at buses (Injection)
            # Note: In pandapower, res_bus.p_mw is the net active power injection (Generation - Load)
            # For State Estimation, we often treat these as "pseudo-measurements" or actual AMI readings.
            # Here we take the net injection at the bus.
            p_meas = net.res_bus.p_mw.values
            q_meas = net.res_bus.q_mvar.values
            
            v_mag_list.append(v_mag)
            v_ang_list.append(v_ang)
            p_meas_list.append(p_meas)
            q_meas_list.append(q_meas)
            
        except pp.LoadflowNotConverged:
            print(f"Warning: Loadflow did not converge at step {t}")
            continue

    # Convert to Tensors
    v_mag_tensor = torch.tensor(np.array(v_mag_list), dtype=torch.float32)
    v_ang_tensor = torch.tensor(np.array(v_ang_list), dtype=torch.float32)
    p_meas_tensor = torch.tensor(np.array(p_meas_list), dtype=torch.float32)
    q_meas_tensor = torch.tensor(np.array(q_meas_list), dtype=torch.float32)
    
    # Save Data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    torch.save({
        'v_mag': v_mag_tensor,
        'v_ang': v_ang_tensor,
        'p_meas': p_meas_tensor,
        'q_meas': q_meas_tensor
    }, os.path.join(output_dir, 'ieee33_time_series.pt'))
    
    # Save Network Structure for Physics Layer extraction later
    # We can pickle the pandapower net, or just rely on reloading case33bw in physics layer
    # But let's save the Y-bus indices if possible. 
    # Actually, simpler to just re-create net in physics layer to extract Y-bus.
    
    print(f"Data generation complete. Saved {len(v_mag_list)} steps to {output_dir}/ieee33_time_series.pt")

if __name__ == "__main__":
    generate_data()
