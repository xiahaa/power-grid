# Benchmark Report: Standard EKF on IEEE 33-Bus System

## 1. Methodology

We implemented a **Standard Extended Kalman Filter (EKF)** to estimate the state of the IEEE 33-bus distribution system. 

*   **State Vector ($x$)**: Voltage magnitudes ($V$) and phase angles ($\theta$) for all 33 buses ($N=66$ state variables).
*   **Measurements ($z$)**: Active ($P$) and Reactive ($Q$) power injections at all buses.
*   **Physics Model**: A Differentiable Power Flow Layer implemented in PyTorch is used as the measurement function $h(x)$. The Jacobian $H_k = \frac{\partial h}{\partial x}$ is computed using `torch.autograd`.
*   **Observability**: To ensure the system is observable (specifically for the reference angle), we added pseudo-measurements for the Slack Bus (Bus 0) with very low variance:
    *   $V_{slack} \approx 1.0$
    *   $\theta_{slack} \approx 0.0$

### 1.1 Experiment Setup
*   **Test System**: IEEE 33-bus radial distribution system.
*   **Data Generation**: 100 time steps of load variations (Random Walk + Daily Cycle) simulated using `pandapower`.
*   **Process Noise ($Q$)**: std = $10^{-4}$ (modeling slow state evolution).
*   **Measurement Noise ($R$)**: std = $10^{-2}$ (approx 1% sensor noise).
*   **Pseudo-Measurement Noise**: std = $10^{-6}$.

## 2. Results

The EKF successfully tracked the voltage state of the system despite the nonlinear load fluctuations.

| Metric | Average RMSE | Unit |
| :--- | :--- | :--- |
| **Voltage Magnitude** | **0.0073** | p.u. |
| **Voltage Angle** | **0.0010** | rad |

### 2.1 Error Analysis
The low RMSE values indicate that the First-Order Linearization (EKF) is sufficient for this level of noise and nonlinearity in the distribution grid. The explicit handling of the slack bus was crucial for stability.

## 3. Visualizations

### 3.1 Tracking Performance
The filter closely follows the ground truth trajectory. Below is the tracking performance for a selected bus (Bus 18).

![Bus Tracking](plots/tracking_bus_18.png)

### 3.2 Estimation Error
The Root Mean Square Error (RMSE) remains bounded and stable over the simulation period.

![RMSE over Time](plots/rmse_over_time.png)

### 3.3 Voltage Profile
A snapshot of the system state at the final time step shows the EKF correctly capturing the voltage drop profile along the feeder.

![Voltage Profile](plots/voltage_profile_snapshot.png)

## 4. Conclusion
This benchmark establishes a strong baseline. The use of a differentiable physics layer allowed for rapid implementation of the EKF by automatically computing Jacobians. Future work will compare this against the proposed **Differentiable Error-State Kalman Filter (DE-KF)** to handle larger topological changes or model mismatches.
