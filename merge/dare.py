import torch
from typing import List, Optional


def dare_drop_and_rescale(
    task_vector: torch.Tensor, 
    drop_rate: float = 0.5,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Drop And Rescale (DARE) operator (Yu et al., 2024).
    Randomly drops parameters with probability drop_rate and rescales surviving
    parameters by 1 / (1 - drop_rate) to maintain expected value.
    """
    assert 0.0 <= drop_rate < 1.0, "drop_rate must be in [0, 1)"
    
    if drop_rate == 0.0:
        return task_vector.clone()
        
    if seed is not None:
        torch.manual_seed(seed)
        
    keep_prob = 1.0 - drop_rate
    mask = torch.bernoulli(torch.full_like(task_vector, keep_prob))
    
    rescaled = (task_vector * mask) / keep_prob
    return rescaled


def dare_merge(
    base_weights: torch.Tensor,
    fine_tuned_weights: List[torch.Tensor],
    drop_rate: float = 0.5,
    scaling_factor: float = 1.0
) -> torch.Tensor:
    """
    Merges multiple fine-tuned models using DARE.
    1. Computes task vectors: tau_m = theta_m - theta_base.
    2. Applies DARE to each task vector (sparsifies + rescales).
    3. Averages sparsified task vectors and adds to base weights.
    """
    assert len(fine_tuned_weights) >= 1, "At least one fine-tuned weight required"
    
    sparsified_task_vectors = []
    for w in fine_tuned_weights:
        tv = w - base_weights
        sparsified_tv = dare_drop_and_rescale(tv, drop_rate=drop_rate)
        sparsified_task_vectors.append(sparsified_tv)
        
    avg_task_vector = torch.mean(torch.stack(sparsified_task_vectors, dim=0), dim=0)
    return base_weights + scaling_factor * avg_task_vector
