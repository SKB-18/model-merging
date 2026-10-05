import math
from typing import List

def slerp(t: float, v0: List[float], v1: List[float]) -> List[float]:
    """
    Spherical Linear Interpolation (SLERP) for two vectors.
    Commonly used for merging language models to preserve properties of weight matrices
    better than simple linear interpolation.
    """
    assert len(v0) == len(v1), "Vectors must be of the same length"
    
    # Compute dot product
    dot = sum(x * y for x, y in zip(v0, v1))
    
    # Calculate magnitudes
    mag_v0 = math.sqrt(sum(x * x for x in v0))
    mag_v1 = math.sqrt(sum(x * x for x in v1))
    
    # Normalize dot product (cosine of the angle)
    if mag_v0 == 0 or mag_v1 == 0:
        return [0.0 for _ in v0]
        
    dot_norm = dot / (mag_v0 * mag_v1)
    
    # Clamp to [-1, 1] to avoid math domain errors
    dot_norm = max(-1.0, min(1.0, dot_norm))
    
    # Angle between vectors
    theta_0 = math.acos(dot_norm)
    
    # If angle is extremely small, fallback to linear interpolation to avoid division by zero
    if theta_0 < 1e-5:
        return [x + t * (y - x) for x, y in zip(v0, v1)]
        
    theta_t = theta_0 * t
    
    # s1 = sin(theta_0 - theta_t) / sin(theta_0)
    s1 = math.sin(theta_0 - theta_t) / math.sin(theta_0)
    
    # s2 = sin(theta_t) / sin(theta_0)
    s2 = math.sin(theta_t) / math.sin(theta_0)
    
    return [s1 * x + s2 * y for x, y in zip(v0, v1)]
