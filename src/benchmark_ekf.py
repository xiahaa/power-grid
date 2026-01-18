import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandapower.networks as pn
import os
from ekf import StandardEKF
from generate_data import generate_data

def run_benchmark():
    # 1. Generate Data (if not exists or force regen)
    print("Generating/Loading Data...")
    if not os.path.exists("data/ieee33_time_series.pt"):
        generate_data()
    
    data = torch.load("data/ieee33_time_series.pt")
    v_mag_gt = data['v_mag']
    v_ang_gt = data['v_ang']
    p_meas = data['p_meas']
    q_meas = data['q_meas']
    
    n_steps = v_mag_gt.shape[0]
    n_bus = v_mag_gt.shape[1]
    
    # 2. Initialize EKF
    print("Initializing EKF...")
    net = pn.case33bw()
    
    # Tuning parameters
    # Process Noise: 1e-4 (Assumes state changes slowly)
    # Measurement Noise: 1e-2 (Assumes ~1% noise roughly, or 0.01 MW/MVar)
    # Note: Measurement noise in generate_data was 0.1 * P_load.
    # Load is ~0.1 MW. So noise is ~0.01 MW.
    # So 1e-2 is a reasonable guess for R.
    ekf = StandardEKF(net, process_noise_std=1e-4, measurement_noise_std=1e-2)
    
    # Storage for results
    est_v_mag = []
    est_v_ang = []
    rmse_mag_list = []
    rmse_ang_list = []
    
    print(f"Running EKF on {n_steps} steps...")
    
    for t in range(n_steps):
        # 1. Predict
        ekf.predict()
        
        # 2. Prepare Measurement Vector
        z_t = torch.cat([p_meas[t], q_meas[t]])
        
        # 3. Update
        ekf.update(z_t)
        
        # 4. Store State
        v_m, v_a = ekf.get_state()
        est_v_mag.append(v_m.detach().numpy())
        est_v_ang.append(v_a.detach().numpy())
        
        # 5. Compute Metrics
        err_mag = v_m - v_mag_gt[t]
        err_ang = v_a - v_ang_gt[t]
        
        rmse_mag = torch.sqrt(torch.mean(err_mag**2)).item()
        rmse_ang = torch.sqrt(torch.mean(err_ang**2)).item()
        
        rmse_mag_list.append(rmse_mag)
        rmse_ang_list.append(rmse_ang)
        
        if t % 10 == 0:
            print(f"Step {t}: RMSE V_mag={rmse_mag:.5f}, RMSE V_ang={rmse_ang:.5f}")
            
    # Convert results to arrays
    est_v_mag = np.array(est_v_mag)
    est_v_ang = np.array(est_v_ang)
    
    # 3. Visualization
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    sns.set_style("whitegrid")
    
    # Plot 1: RMSE over time
    plt.figure(figsize=(10, 5))
    plt.plot(rmse_mag_list, label='Voltage Magnitude RMSE (p.u.)')
    plt.plot(rmse_ang_list, label='Voltage Angle RMSE (rad)')
    plt.title('EKF Estimation Error over Time')
    plt.xlabel('Time Step')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig("plots/rmse_over_time.png")
    plt.close()
    
    # Plot 2: Single Bus Tracking (e.g., Bus 18, end of a feeder)
    bus_idx = 17 # Bus 18 in 0-indexed
    plt.figure(figsize=(10, 5))
    plt.plot(v_mag_gt[:, bus_idx].numpy(), 'k--', label='True Magnitude')
    plt.plot(est_v_mag[:, bus_idx], 'r-', label='Estimated Magnitude')
    plt.title(f'Voltage Magnitude Tracking at Bus {bus_idx+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Voltage (p.u.)')
    plt.legend()
    plt.savefig(f"plots/tracking_bus_{bus_idx+1}.png")
    plt.close()
    
    # Plot 3: Voltage Profile Snapshot at last step
    plt.figure(figsize=(10, 5))
    plt.plot(v_mag_gt[-1].numpy(), 'bo-', label='True Profile')
    plt.plot(est_v_mag[-1], 'rs--', label='Estimated Profile')
    plt.title(f'System Voltage Profile (Step {n_steps})')
    plt.xlabel('Bus Index')
    plt.ylabel('Voltage (p.u.)')
    plt.legend()
    plt.savefig("plots/voltage_profile_snapshot.png")
    plt.close()

    print("Benchmark Complete. Plots saved to plots/ directory.")
    
    # Return average metrics for report
    avg_rmse_mag = np.mean(rmse_mag_list)
    avg_rmse_ang = np.mean(rmse_ang_list)
    return avg_rmse_mag, avg_rmse_ang

if __name__ == "__main__":
    avg_mag, avg_ang = run_benchmark()
    print(f"Average RMSE Mag: {avg_mag:.6f}")
    print(f"Average RMSE Ang: {avg_ang:.6f}")
