import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        # Create constant PE matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, Max_Len, D]
        
    def forward(self, x):
        """
        x: [Batch, Seq_Len, D]
        """
        return x + self.pe[:, :x.size(1), :]

class ErrorStateTransformer(nn.Module):
    """
    Predicts the Error State correction (delta_x) and Process Noise (Q_diag)
    based on measurement history using a Transformer Encoder.
    """
    def __init__(self, n_meas, n_state, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Linear(n_meas, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Heads
        # We process the sequence and take the embedding of the *last* token to predict correction
        self.fc_mean = nn.Linear(d_model, n_state)
        self.fc_logvar = nn.Linear(d_model, n_state)
        
    def forward(self, z_seq):
        """
        z_seq: [Batch, Seq_Len, n_meas]
        Returns: 
            delta_x: [Batch, n_state]
            Q_diag: [Batch, n_state]
        """
        # Embed
        x = self.embedding(z_seq) # [Batch, Seq, D]
        x = self.pos_encoder(x)
        
        # Transform
        # Output: [Batch, Seq, D]
        # We don't need a mask for causal attention if we just look at the whole history window 
        # to predict the CURRENT step correction. The history is already "past".
        encoded = self.transformer_encoder(x)
        
        # Take the last token representation (corresponding to most recent measurement)
        last_token = encoded[:, -1, :]
        
        delta_x = self.fc_mean(last_token)
        logvar_Q = self.fc_logvar(last_token)
        
        Q_diag = torch.exp(logvar_Q)
        
        return delta_x, Q_diag

class MeasurementNoiseNet(nn.Module):
    """
    HyperNetwork that predicts Measurement Noise Covariance (R) 
    based on current measurement.
    """
    def __init__(self, n_meas, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_meas, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_meas) # Diagonal R
        )
        
    def forward(self, z):
        """
        z: [Batch, n_meas]
        Returns: R_diag: [Batch, n_meas]
        """
        logvar_R = self.net(z)
        R_diag = torch.exp(logvar_R)
        return R_diag
