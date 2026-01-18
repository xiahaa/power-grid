import torch
import torch.nn as nn
from physics import PowerFlowLayer

class StandardEKF(nn.Module):
    def __init__(self, net, process_noise_std=1e-4, measurement_noise_std=1e-2):
        super().__init__()
        self.pf_layer = PowerFlowLayer(net)
        self.n_bus = self.pf_layer.Ybus.shape[0]
        self.n_state = 2 * self.n_bus  # [v_mag, v_ang]
        
        # Standard Measurements: P, Q at all buses (2*N)
        # Pseudo Measurements: V_slack, Theta_slack (2)
        # Total Measurements: 2*N + 2
        self.n_meas = 2 * self.n_bus + 2
        
        # Initialize Covariance Matrices
        # Process Noise Q: assume uncorrelated noise on state evolution
        self.Q = torch.eye(self.n_state) * (process_noise_std ** 2)
        
        # Measurement Noise R
        # For standard measurements: measurement_noise_std
        # For pseudo measurements: very small noise (trust them highly)
        self.R = torch.eye(self.n_meas) * (measurement_noise_std ** 2)
        
        # Pseudo-measurement noise (Slack Bus is known perfectly)
        pseudo_meas_std = 1e-6
        self.R[-2, -2] = pseudo_meas_std ** 2 # V_slack
        self.R[-1, -1] = pseudo_meas_std ** 2 # Theta_slack
        
        # State Estimate Covariance P
        self.P = torch.eye(self.n_state) * 0.1 
        
        # Initial State Estimate (Flat start)
        self.x = torch.zeros(self.n_state)
        self.x[:self.n_bus] = 1.0 # Voltages = 1.0
        # Angles = 0.0
        
    def predict(self):
        """
        Prediction Step (Time Update).
        Assumes quasi-static state (Random Walk): x_k = x_{k-1} + w_k
        """
        self.P = self.P + self.Q
        
    def measurement_function(self, x):
        """
        Wrapper for PowerFlowLayer + Pseudo Measurements.
        x: [v_mag, v_ang] (Size: 2*N)
        Returns: z_pred: [p_inj, q_inj, v_slack, theta_slack] (Size: 2*N + 2)
        """
        v_mag = x[:self.n_bus]
        v_ang = x[self.n_bus:]
        
        # Add batch dimension for PowerFlowLayer
        v_mag_batch = v_mag.unsqueeze(0)
        v_ang_batch = v_ang.unsqueeze(0)
        
        p_inj, q_inj = self.pf_layer(v_mag_batch, v_ang_batch)
        
        # Standard Measurements
        z_standard = torch.cat([p_inj.squeeze(0), q_inj.squeeze(0)])
        
        # Pseudo Measurements (Bus 0 is Slack)
        # V_slack = v_mag[0]
        # Theta_slack = v_ang[0]
        z_pseudo = torch.stack([v_mag[0], v_ang[0]])
        
        z_pred = torch.cat([z_standard, z_pseudo])
        return z_pred

    def update(self, z_meas_standard):
        """
        Correction Step (Measurement Update).
        z_meas_standard: [p_inj, q_inj] (Size: 2*N)
        """
        # Append Pseudo-Measurements to the measurement vector
        # We assume Slack Voltage is 1.0 and Slack Angle is 0.0
        z_pseudo = torch.tensor([1.0, 0.0], device=z_meas_standard.device)
        z_meas = torch.cat([z_meas_standard, z_pseudo])
        
        # 1. Compute Jacobian H = dh/dx at current state x
        H = torch.autograd.functional.jacobian(self.measurement_function, self.x)
        
        # 2. Compute Innovation (Residual) y = z_meas - h(x)
        z_pred = self.measurement_function(self.x)
        y = z_meas - z_pred
        
        # 3. Compute Innovation Covariance S = H P H^T + R
        S = H @ self.P @ H.T + self.R
        
        # 4. Compute Kalman Gain K = P H^T S^{-1}
        # Use torch.linalg.solve(S, H @ P).T for stability (S K^T = H P) -> K = (S^-1 H P)^T = P H^T S^-T = P H^T S^-1
        # Better: S @ K^T = (H @ P) -> K^T = solve(S, H@P) -> K = solve(S, H@P).T
        # Note: P is symmetric, S is symmetric.
        # K = P H^T S^-1
        
        # Compute K using solve
        # K = (S^{-1} @ (H @ P)).T ? No.
        # K = (P @ H.T) @ S^{-1}
        # Let A = S, B = (P @ H.T). We want B @ A^{-1}.
        # This is (A^{-T} @ B^T)^T = (solve(A.T, B.T)).T
        # Since S is symmetric, A=A.T.
        # K = (solve(S, (P @ H.T).T)).T = (solve(S, H @ P)).T
        
        numerator = H @ self.P # Shape [Meas, State]
        # We need K [State, Meas].
        # K = P H^T S^-1
        # K^T = S^-1 H P
        # S K^T = H P
        # K^T = solve(S, H @ P)
        
        K_T = torch.linalg.solve(S, numerator)
        K = K_T.T
        
        # 5. Update State x = x + K y
        self.x = self.x + K @ y
        
        # Clamp Voltage Magnitudes to be positive to avoid physics issues
        # (Though EKF should keep them near 1.0)
        self.x[:self.n_bus] = torch.clamp(self.x[:self.n_bus], min=0.1)
        
        # 6. Update Covariance P = (I - K H) P
        # Using Joseph form for stability: P = (I - KH) P (I - KH)^T + K R K^T
        # But standard form is faster: P = P - K H P = P - K @ numerator
        self.P = self.P - K @ numerator
        
        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)
        
        return self.x.clone()

    def get_state(self):
        """
        Returns separated v_mag and v_ang
        """
        return self.x[:self.n_bus], self.x[self.n_bus:]
