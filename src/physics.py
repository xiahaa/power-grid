import torch
import torch.nn as nn
import pandapower as pp
import pandapower.networks as pn
import numpy as np
import scipy.sparse as sp
from pandapower.pypower.makeYbus import makeYbus

def get_ybus(net):
    """
    Extracts the complex Y-bus matrix from a pandapower network.
    Returns:
        Y_bus (torch.complex64): The admittance matrix (N_bus x N_bus).
    """
    # Create internal ppc (PYPOWER case) if not already present
    # We need to run runpp to ensure the ppc structure is fully initialized and converted
    # Or explicitly call pd2ppc
    try:
        if net._ppc is None:
             pp.runpp(net)
    except:
        pp.runpp(net)
        
    # Pandapower stores the internal pypower case in net._ppc
    ppc = net._ppc
    
    # Use internal pypower function to build Ybus
    # makeYbus returns sparse matrix (csr_matrix or similar)
    Ybus_csc, Yf, Yt = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])
    
    # Convert to dense numpy then torch
    # Note: For large systems, we should use sparse tensors, 
    # but for IEEE 33 (33 nodes), dense is fine and faster for dev.
    Ybus_np = Ybus_csc.toarray()
    
    Ybus_torch = torch.tensor(Ybus_np, dtype=torch.cfloat)
    return Ybus_torch

class PowerFlowLayer(nn.Module):
    """
    Differentiable Physics Layer that computes Power Injections (P, Q) 
    given Voltage State (Mag, Ang) and System Admittance (Ybus).
    
    Input:
        v_mag: (Batch, N_bus) - Voltage Magnitudes (p.u.)
        v_ang: (Batch, N_bus) - Voltage Angles (radians)
        
    Output:
        p_inj: (Batch, N_bus) - Active Power Injection (MW)
        q_inj: (Batch, N_bus) - Reactive Power Injection (MVar)
    """
    def __init__(self, net):
        super().__init__()
        # Extract Ybus from the network
        self.Ybus = get_ybus(net)
        
        # Get baseMVA for p.u. conversion
        if net._ppc is not None and 'baseMVA' in net._ppc:
             self.baseMVA = float(net._ppc['baseMVA'])
        else:
             self.baseMVA = 100.0 
             
        # Register Ybus as a buffer
        self.register_buffer('G', self.Ybus.real)
        self.register_buffer('B', self.Ybus.imag)
        
    def forward(self, v_mag, v_ang):
        """
        Computes P and Q using the power flow equations in matrix form.
        
        Complex Voltage V = v_mag * e^(j * v_ang)
        Current I = Y * V
        Power S = V * conj(I) = P + jQ
        """
        # Create Complex Voltage Tensor
        # v_mag: [Batch, N]
        # v_ang: [Batch, N]
        
        # Euler's formula: V = |V| (cos(theta) + j sin(theta))
        v_real = v_mag * torch.cos(v_ang)
        v_imag = v_mag * torch.sin(v_ang)
        V = torch.complex(v_real, v_imag) # [Batch, N]
        
        # Compute Current Injection: I = Y * V
        # Y is [N, N], V is [Batch, N]. We need (V @ Y^T) or equivalent.
        # Since Y is symmetric (mostly), but strictly I = Y @ V for a single vector.
        # For batch: I_batch = (Y @ V_batch.T).T = V_batch @ Y.T
        
        # Note: Ybus in pandapower is usually symmetric for lines, but can be asymmetric with phase shifters.
        # It's safer to use Y @ V formulation.
        
        # V: [Batch, N] -> [Batch, N, 1]
        V_unsqueezed = V.unsqueeze(-1)
        
        # Y: [N, N] -> [1, N, N] (broadcast over batch)
        Y_batch = torch.complex(self.G, self.B).unsqueeze(0)
        
        # I = Y * V
        # [Batch, N, N] @ [Batch, N, 1] -> [Batch, N, 1]
        # But we need to be careful with broadcasting.
        
        # Let's do: I = (V @ Y.T)
        I = torch.matmul(V, torch.complex(self.G, self.B).T)
        
        # Complex Power S = V * conj(I) (Element-wise multiplication)
        S = V * torch.conj(I)
        
        # P = Real(S), Q = Imag(S) in per unit
        p_inj_pu = S.real
        q_inj_pu = S.imag
        
        # Convert to Physical Units (MW, MVar)
        # Pandapower convention for res_bus.p_mw is Generation - Load
        # Our physics layer calculates Injection (S = V I*). 
        # However, Pandapower results seem to treat load buses as positive P if they are consuming?
        # Let's check:
        # Bus 1 has Load 0.1 MW. res_bus.p_mw is 0.1.
        # Bus 0 has ExtGrid injecting 3.9 MW. res_bus.p_mw is -3.9.
        
        # Wait, usually Injection = Gen - Load.
        # If Load = 0.1, Gen = 0, Injection = -0.1.
        # But pandapower res_bus.p_mw says 0.1.
        # This implies res_bus.p_mw = Load - Gen ? Or Net Consumption?
        # Documentation says: "p_mw (float) - Active power demand"
        # Ah! `res_bus.p_mw` is NOT net injection. It is the RESULTING power at the bus?
        # No, let's check docs or behavior.
        # Bus 0 (Slack): Gen=3.9, Load=0. res_bus.p_mw = -3.9.
        # This means res_bus.p_mw is defined as (Load - Gen).
        # Positive means acting as a Load. Negative means acting as a Generator.
        
        # My physics layer calculates S = V I* = P_inj + jQ_inj.
        # This is standard injection (Gen - Load).
        # So P_physics = (Gen - Load).
        # Pandapower P_res = (Load - Gen).
        # So P_physics = - P_res.
        
        p_inj = p_inj_pu * self.baseMVA
        q_inj = q_inj_pu * self.baseMVA
        
        # Negate to match Pandapower's Load-Gen convention
        return -p_inj, -q_inj
