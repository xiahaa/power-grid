import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import torch
import os

def generate_robust_data(output_dir="data", n_seqs=20, seq_len=100, mode='train'):
    """
    Generates robust datasets with dynamic load changes.
    n_seqs: Number of separate time-series trajectories.
    seq_len: Length of each trajectory.
    """
    print(f"Generating {mode} data: {n_seqs} sequences of length {seq_len}...")
    
    net = pn.case33bw()
    n_bus = len(net.bus)
    
    # Storage
    all_v_mag = []
    all_v_ang = []
    all_p_meas = []
    all_q_meas = []
    
    # Base Loads
    p_load_base = net.load.p_mw.values.copy()
    q_load_base = net.load.q_mvar.values.copy()
    
    for s in range(n_seqs):
        # Initialize sequence storage
        seq_v_mag = []
        seq_v_ang = []
        seq_p_meas = []
        seq_q_meas = []
        
        # Randomize initial load level (0.8 to 1.2 pu)
        load_scale = np.random.uniform(0.8, 1.2)
        
        # Current loads
        current_p = p_load_base * load_scale
        current_q = q_load_base * load_scale
        
        # Dynamic Pattern Selection
        pattern_type = np.random.choice(['ramp', 'step', 'random_walk'])
        
        for t in range(seq_len):
            # Apply Dynamics
            if pattern_type == 'random_walk':
                # Slow drift
                noise = np.random.normal(0, 0.05, size=len(p_load_base))
                current_p *= (1 + noise)
                current_q *= (1 + noise)
            elif pattern_type == 'ramp':
                # Constant ramp up or down
                ramp_rate = 1.0 + np.random.choice([-0.02, 0.02])
                current_p *= ramp_rate
                current_q *= ramp_rate
            elif pattern_type == 'step':
                # Sudden jump at random time
                if t == seq_len // 2:
                    jump = np.random.uniform(0.7, 1.3)
                    current_p *= jump
                    current_q *= jump
                else:
                    # Small noise
                    noise = np.random.normal(0, 0.01, size=len(p_load_base))
                    current_p *= (1 + noise)
                    current_q *= (1 + noise)
            
            # Update network
            net.load.p_mw = current_p
            net.load.q_mvar = current_q
            
            try:
                pp.runpp(net)
                
                # Ground Truth
                v_mag = net.res_bus.vm_pu.values.copy()
                v_ang = np.deg2rad(net.res_bus.va_degree.values.copy())
                
                # Measurements (Noisy)
                # True Injection (Gen - Load)
                # Pandapower res_bus.p_mw is Load - Gen. So we negate it.
                p_true = -net.res_bus.p_mw.values
                q_true = -net.res_bus.q_mvar.values
                
                # Add Measurement Noise (1% standard)
                p_meas = p_true + np.random.normal(0, 0.01, size=n_bus)
                q_meas = q_true + np.random.normal(0, 0.01, size=n_bus)
                
                seq_v_mag.append(v_mag)
                seq_v_ang.append(v_ang)
                seq_p_meas.append(p_meas)
                seq_q_meas.append(q_meas)
                
            except pp.LoadflowNotConverged:
                if len(seq_v_mag) > 0:
                    seq_v_mag.append(seq_v_mag[-1])
                    seq_v_ang.append(seq_v_ang[-1])
                    seq_p_meas.append(seq_p_meas[-1])
                    seq_q_meas.append(seq_q_meas[-1])
                else:
                    pass

        # Pad if short
        while len(seq_v_mag) < seq_len:
             seq_v_mag.append(seq_v_mag[-1])
             seq_v_ang.append(seq_v_ang[-1])
             seq_p_meas.append(seq_p_meas[-1])
             seq_q_meas.append(seq_q_meas[-1])
             
        all_v_mag.append(seq_v_mag)
        all_v_ang.append(seq_v_ang)
        all_p_meas.append(seq_p_meas)
        all_q_meas.append(seq_q_meas)

    # Convert to Tensors
    data = {
        'v_mag': torch.tensor(np.array(all_v_mag), dtype=torch.float32),
        'v_ang': torch.tensor(np.array(all_v_ang), dtype=torch.float32),
        'p_meas': torch.tensor(np.array(all_p_meas), dtype=torch.float32),
        'q_meas': torch.tensor(np.array(all_q_meas), dtype=torch.float32)
    }
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    torch.save(data, os.path.join(output_dir, f'robust_{mode}.pt'))
    print(f"Saved {output_dir}/robust_{mode}.pt")

if __name__ == "__main__":
    generate_robust_data(n_seqs=50, seq_len=50, mode='train')
    generate_robust_data(n_seqs=10, seq_len=50, mode='test')
