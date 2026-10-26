import math
from typing import List, Union
import torch


def slerp(t: float, v0: List[float], v1: List[float]) -> List[float]:
    """
    Spherical Linear Interpolation (SLERP) for two 1D lists.
    Preserves vector magnitude on the unit hypersphere.
    """
    assert len(v0) == len(v1), "Vectors must be of the same length"
    
    dot = sum(x * y for x, y in zip(v0, v1))
    mag_v0 = math.sqrt(sum(x * x for x in v0))
    mag_v1 = math.sqrt(sum(x * x for x in v1))
    
    if mag_v0 == 0 or mag_v1 == 0:
        return [0.0 for _ in v0]
        
    dot_norm = max(-1.0, min(1.0, dot / (mag_v0 * mag_v1)))
    theta_0 = math.acos(dot_norm)
    
    # Fallback to linear interpolation if angle is near collinear
    if theta_0 < 1e-5:
        return [x + t * (y - x) for x, y in zip(v0, v1)]
        
    theta_t = theta_0 * t
    s1 = math.sin(theta_0 - theta_t) / math.sin(theta_0)
    s2 = math.sin(theta_t) / math.sin(theta_0)
    
    return [s1 * x + s2 * y for x, y in zip(v0, v1)]


def slerp_tensor(t: float, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
    """
    PyTorch Tensor-level Spherical Linear Interpolation (SLERP).
    Applies SLERP across arbitrary weight matrices and model layer tensors.
    """
    assert t0.shape == t1.shape, f"Tensors must match shape, got {t0.shape} vs {t1.shape}"
    
    original_shape = t0.shape
    v0 = t0.flatten()
    v1 = t1.flatten()
    
    norm_v0 = torch.norm(v0)
    norm_v1 = torch.norm(v1)
    
    if norm_v0 < 1e-8 or norm_v1 < 1e-8:
        return torch.lerp(t0, t1, t)
        
    u0 = v0 / norm_v0
    u1 = v1 / norm_v1
    
    dot = torch.sum(u0 * u1).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    
    if theta < 1e-5:
        return torch.lerp(t0, t1, t)
        
    sin_theta = torch.sin(theta)
    s0 = torch.sin((1.0 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    
    merged_flat = s0 * v0 + s1 * v1
    return merged_flat.reshape(original_shape)
