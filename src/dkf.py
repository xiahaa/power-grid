import torch
import torch.nn as nn
from physics import PowerFlowLayer
from manifold import boxplus, boxminus
from models import ErrorStateTransformer, MeasurementNoiseNet

class DifferentiableESKF(nn.Module):
    def __init__(self, net, n_history=5):
        super().__init__()
        self.pf_layer = PowerFlowLayer(net)
        self.n_bus = self.pf_layer.Ybus.shape[0]
        self.n_state = 2 * self.n_bus
        self.n_meas = 2 * self.n_bus + 2 # P, Q + Pseudo Slack
        
        self.n_history = n_history
        
        # Neural Components (Updated to Transformer)
        self.error_net = ErrorStateTransformer(n_meas=self.n_meas, n_state=self.n_state, d_model=64)
        self.meas_noise_net = MeasurementNoiseNet(n_meas=self.n_meas)
        
        # Initial P
        self.P0_logvar = nn.Parameter(torch.ones(self.n_state) * -2.0)
        
    def measurement_function(self, x):
        """
        Differentiable Measurement Function h(x).
        x: [Batch, n_state]
        """
        v_mag = x[:, :self.n_bus]
        v_ang = x[:, self.n_bus:]
        
        p_inj, q_inj = self.pf_layer(v_mag, v_ang) # [Batch, N]
        
        # Pseudo measurements for slack bus (Bus 0)
        z_pseudo = torch.stack([v_mag[:, 0], v_ang[:, 0]], dim=1) # [Batch, 2]
        
        z_pred = torch.cat([p_inj, q_inj, z_pseudo], dim=1) # [Batch, 2N+2]
        return z_pred

    def forward(self, z_seq_batch, x_init=None, P_init=None):
        """
        Run Filter over a sequence.
        z_seq_batch: [Batch, Seq_Len, n_meas]
        """
        batch_size, seq_len, _ = z_seq_batch.shape
        device = z_seq_batch.device
        
        if x_init is None:
            x_curr = torch.zeros(batch_size, self.n_state, device=device)
            x_curr[:, :self.n_bus] = 1.0 # Flat start
        else:
            x_curr = x_init
            
        if P_init is None:
            P_curr = torch.diag_embed(torch.exp(self.P0_logvar)).repeat(batch_size, 1, 1)
        else:
            P_curr = P_init
            
        x_est_seq = []
        
        # Measurement history buffer
        z_buffer = torch.zeros(batch_size, self.n_history, self.n_meas, device=device)
        
        for k in range(seq_len):
            z_k = z_seq_batch[:, k, :]
            
            # Update history buffer
            z_buffer = torch.cat([z_buffer[:, 1:, :], z_k.unsqueeze(1)], dim=1)
            
            # --- 1. PREDICTION STEP ---
            x_nom = x_curr 
            
            # Neural Error Prediction (Transformer)
            delta_x, Q_diag = self.error_net(z_buffer) 
            
            x_pred = boxplus(x_nom, delta_x)
            
            Q = torch.diag_embed(Q_diag)
            P_pred = P_curr + Q
            
            # --- 2. UPDATE STEP ---
            z_pred = self.measurement_function(x_pred)
            
            R_diag = self.meas_noise_net(z_k)
            R = torch.diag_embed(R_diag)
            
            # Jacobian H
            H_list = []
            for b in range(batch_size):
                def func_single(x_vec):
                    return self.measurement_function(x_vec.unsqueeze(0)).squeeze(0)
                jac = torch.autograd.functional.jacobian(func_single, x_pred[b])
                H_list.append(jac)
            H = torch.stack(H_list)
            
            # Kalman Gain
            S = H @ P_pred @ H.transpose(1, 2) + R
            numerator = H @ P_pred
            K_T = torch.linalg.solve(S, numerator)
            K = K_T.transpose(1, 2)
            
            # Update
            y = z_k - z_pred
            dx_update = (K @ y.unsqueeze(-1)).squeeze(-1)
            x_est = boxplus(x_pred, dx_update)
            
            I = torch.eye(self.n_state, device=device).unsqueeze(0)
            P_est = (I - K @ H) @ P_pred
            
            x_est_seq.append(x_est)
            
            x_curr = x_est
            P_curr = P_est
            
        return torch.stack(x_est_seq, dim=1)
