import torch
from typing import List, Dict, Tuple, Optional


def trim_task_vector(task_vector: torch.Tensor, top_k_fraction: float = 0.2) -> torch.Tensor:
    """
    Step 1: Trims task vector to retain only the top_k_fraction parameters by absolute magnitude.
    Sets all remaining parameters to zero.
    """
    assert 0.0 < top_k_fraction <= 1.0, "top_k_fraction must be in (0, 1]"
    flat = task_vector.flatten()
    k = max(1, int(len(flat) * top_k_fraction))
    
    threshold = torch.topk(torch.abs(flat), k).values[-1]
    mask = torch.abs(task_vector) >= threshold
    return task_vector * mask


def elect_signs(trimmed_task_vectors: List[torch.Tensor]) -> torch.Tensor:
    """
    Step 2: Elects the consensus sign direction across multiple task vectors.
    Returns +1, -1, or 0 for each parameter.
    """
    stacked = torch.stack(trimmed_task_vectors, dim=0)
    sum_vectors = torch.sum(stacked, dim=0)
    return torch.sign(sum_vectors)


def disjoint_merge(
    trimmed_task_vectors: List[torch.Tensor], 
    elected_signs: torch.Tensor
) -> torch.Tensor:
    """
    Step 3: Averages parameters only from task vectors that agree with the elected sign.
    """
    stacked = torch.stack(trimmed_task_vectors, dim=0)
    # Mask of agreeing parameters: sign(tau_m) == elected_sign and tau_m != 0
    signs_match = (torch.sign(stacked) == elected_signs.unsqueeze(0)) & (stacked != 0)
    
    agreeing_values = stacked * signs_match
    agreeing_counts = torch.sum(signs_match, dim=0).clamp(min=1)
    
    sum_agreeing = torch.sum(agreeing_values, dim=0)
    merged_vector = sum_agreeing / agreeing_counts
    
    # Where no task vector agreed (count == 0), set to zero
    has_agreement = torch.sum(signs_match, dim=0) > 0
    return torch.where(has_agreement, merged_vector, torch.zeros_like(merged_vector))


def ties_merge(
    base_weights: torch.Tensor,
    fine_tuned_weights: List[torch.Tensor],
    top_k_fraction: float = 0.2,
    scaling_factor: float = 1.0
) -> torch.Tensor:
    """
    Complete TIES-Merging Pipeline (Yadav et al., NeurIPS 2023).
    1. Computes task vectors: tau_m = theta_m - theta_base.
    2. Trims to top-k% magnitude.
    3. Elects consensus signs to eliminate interference.
    4. Disjoint merges agreeing parameter directions.
    5. Recombines with base weights: theta_merged = theta_base + lambda * tau_ties.
    """
    assert len(fine_tuned_weights) >= 1, "At least one fine-tuned model weight required"
    
    # 1. Compute task vectors
    task_vectors = [w - base_weights for w in fine_tuned_weights]
    
    # 2. Trim
    trimmed = [trim_task_vector(tv, top_k_fraction) for tv in task_vectors]
    
    # 3. Elect Sign
    consensus_signs = elect_signs(trimmed)
    
    # 4. Disjoint Merge
    merged_task_vector = disjoint_merge(trimmed, consensus_signs)
    
    # 5. Recombine with base
    return base_weights + scaling_factor * merged_task_vector
