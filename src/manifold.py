import torch

def boxplus(x, delta):
    """
    Manifold 'Addition' (Retraction).
    x: [Batch, 2*N] (v_mag, v_ang)
    delta: [Batch, 2*N] (delta_v_mag, delta_v_ang)
    
    Returns: x_new on the manifold.
    """
    n_bus = x.shape[-1] // 2
    
    v_mag = x[..., :n_bus]
    v_ang = x[..., n_bus:]
    
    delta_v_mag = delta[..., :n_bus]
    delta_v_ang = delta[..., n_bus:]
    
    # Voltage Magnitude: Euclidean addition (but strictly positive)
    # We can enforce positivity via exp map or simple clamp, or just addition.
    # Standard EKF uses addition. Let's use addition but clamp implies boundary.
    # Ideally: v_new = v * exp(delta_v) to guarantee positivity? 
    # Or just v_new = v + delta_v. For ESKF, usually v + delta.
    v_mag_new = v_mag + delta_v_mag
    
    # Voltage Angle: Add and wrap to [-pi, pi) or [0, 2pi)
    # This is the SO(2) / S1 manifold part.
    v_ang_new = v_ang + delta_v_ang
    # Wrap to [-pi, pi]
    v_ang_new = torch.remainder(v_ang_new + torch.pi, 2 * torch.pi) - torch.pi
    
    return torch.cat([v_mag_new, v_ang_new], dim=-1)

def boxminus(x1, x2):
    """
    Manifold 'Subtraction' (Log map / Error).
    Returns delta = x1 [-] x2 in the tangent space of x2.
    """
    n_bus = x1.shape[-1] // 2
    
    v_mag1 = x1[..., :n_bus]
    v_ang1 = x1[..., n_bus:]
    
    v_mag2 = x2[..., :n_bus]
    v_ang2 = x2[..., n_bus:]
    
    # Magnitude diff
    diff_mag = v_mag1 - v_mag2
    
    # Angle diff (Shortest path on circle)
    diff_ang = v_ang1 - v_ang2
    # Normalize to [-pi, pi]
    diff_ang = torch.remainder(diff_ang + torch.pi, 2 * torch.pi) - torch.pi
    
    return torch.cat([diff_mag, diff_ang], dim=-1)
