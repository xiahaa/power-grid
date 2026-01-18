import torch
import pandapower.networks as pn
import numpy as np
import os
from physics import PowerFlowLayer

def verify():
    print("Loading generated data...")
    data_path = "data/ieee33_time_series.pt"
    if not os.path.exists(data_path):
        print("Error: Data file not found. Run generate_data.py first.")
        return

    data = torch.load(data_path)
    v_mag_gt = data['v_mag']
    v_ang_gt = data['v_ang']
    p_meas_gt = data['p_meas']
    q_meas_gt = data['q_meas']
    
    print(f"Loaded {v_mag_gt.shape[0]} samples.")
    
    print("Initializing PowerFlowLayer with IEEE 33-bus...")
    net = pn.case33bw()
    pf_layer = PowerFlowLayer(net)
    
    # Move to GPU if available? (Keep CPU for verification simplicity)
    
    print("Running Differentiable Physics Layer...")
    # Forward pass: Calculate P, Q from V_GT, Theta_GT
    with torch.no_grad():
        p_calc, q_calc = pf_layer(v_mag_gt, v_ang_gt)
        
    # Compare with "Measured" P, Q (which were calculated by pandapower in data generation)
    # Note: data generation stored 'net.res_bus.p_mw' which is (Generation - Load).
    # PowerFlowLayer calculates Injection S = V * conj(I).
    # Injection should match P_meas.
    
    # Calculate Error
    p_error = torch.abs(p_calc - p_meas_gt)
    q_error = torch.abs(q_calc - q_meas_gt)
    
    max_p_error = torch.max(p_error).item()
    max_q_error = torch.max(q_error).item()
    mse_p_error = torch.mean(p_error**2).item()
    
    print(f"\nVerification Results:")
    print(f"Max Active Power Error (MW): {max_p_error:.6f}")
    print(f"Max Reactive Power Error (MVar): {max_q_error:.6f}")
    print(f"MSE Active Power Error: {mse_p_error:.8f}")
    
    # Thresholds (Numerical precision differences are expected between torch and internal pandapower solver)
    # Pandapower uses float64 usually, Torch float32 by default here.
    tolerance = 1e-4 # 0.1 kW
    
    if max_p_error < tolerance and max_q_error < tolerance:
        print("\nSUCCESS: PyTorch Physics Layer matches Pandapower results!")
    else:
        print("\nWARNING: Discrepancy detected.")
        print("Possible reasons: Float32 vs Float64 precision, or sign convention mismatch.")
        print("Note: Pandapower P_bus = P_gen - P_load.")
        print("Physics Layer calculates P_injection directly.")
        
        # Debugging one sample if needed
        # print("Sample 0 P_calc:", p_calc[0, :5])
        # print("Sample 0 P_meas:", p_meas_gt[0, :5])

if __name__ == "__main__":
    verify()
