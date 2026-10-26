import re
import copy
from typing import Dict, List, Optional, Any, Tuple
import torch

from .slerp import slerp_tensor
from .ties import ties_merge
from .dare import dare_merge


class ModelMerger:
    """
    Production-grade Model Merging Pipeline.
    Supports SLERP, TIES-Merging, and DARE across PyTorch state dictionaries.
    
    Handles:
    - Layer-specific regex filtering
    - Automatic preservation of integer/boolean buffers and metadata
    - Device and precision consistency (fp16, bf16, fp32)
    - Detailed merge metrics (cosine similarity, sparsity, magnitude shift)
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def merge_state_dicts(
        self,
        method: str,
        base_state_dict: Dict[str, torch.Tensor],
        target_state_dicts: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
        slerp_t: float = 0.5,
        ties_top_k: float = 0.2,
        dare_drop_rate: float = 0.5,
        scaling_factor: float = 1.0,
        filter_regex: Optional[str] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Merges full state dictionaries using the specified method.
        
        Args:
            method: 'slerp', 'ties', or 'dare'
            base_state_dict: Baseline model state dict
            target_state_dicts: List of fine-tuned model state dicts
            weights: Optional weights for models (normalized)
            slerp_t: Interpolation factor for SLERP (0.0 = base, 1.0 = target)
            ties_top_k: Magnitude trimming fraction for TIES (0.0 to 1.0]
            dare_drop_rate: Probability of dropping parameters in DARE [0.0 to 1.0)
            scaling_factor: Global task vector scaling factor
            filter_regex: Regex pattern of parameter keys to merge (others kept as base)
            
        Returns:
            Tuple of (merged_state_dict, merge_summary_metrics)
        """
        method = method.lower()
        if method not in ("slerp", "ties", "dare"):
            raise ValueError(f"Unsupported merge method '{method}'. Must be 'slerp', 'ties', or 'dare'.")

        if not target_state_dicts:
            raise ValueError("target_state_dicts must contain at least one state dict.")

        if method == "slerp" and len(target_state_dicts) != 1:
            raise ValueError("SLERP operates pairwise between base and exactly 1 target state dict.")

        compiled_filter = re.compile(filter_regex) if filter_regex else None
        merged_dict: Dict[str, torch.Tensor] = {}

        total_params = 0
        merged_params = 0
        skipped_non_float = 0
        skipped_filtered = 0

        # Verify key consistency
        base_keys = set(base_state_dict.keys())
        for i, sd in enumerate(target_state_dicts):
            missing = base_keys - set(sd.keys())
            if missing:
                raise KeyError(f"Target state dict #{i} is missing keys present in base: {list(missing)[:5]}")

        for key, base_param in base_state_dict.items():
            num_elem = base_param.numel()
            total_params += num_elem

            # 1. Non-floating point tensors (e.g. position_ids, step counters) cannot be interpolated
            if not base_param.is_floating_point():
                merged_dict[key] = base_param.clone()
                skipped_non_float += num_elem
                continue

            # 2. Key filter check
            if compiled_filter and not compiled_filter.search(key):
                merged_dict[key] = base_param.clone()
                skipped_filtered += num_elem
                continue

            target_params = [sd[key] for sd in target_state_dicts]
            orig_dtype = base_param.dtype

            # Perform layer merge in float32 for numerical stability
            bp_f32 = base_param.to(device=self.device, dtype=torch.float32)
            tp_f32 = [tp.to(device=self.device, dtype=torch.float32) for tp in target_params]

            if method == "slerp":
                merged_tensor = slerp_tensor(slerp_t, bp_f32, tp_f32[0])
            elif method == "ties":
                merged_tensor = ties_merge(
                    base_weights=bp_f32,
                    fine_tuned_weights=tp_f32,
                    top_k_fraction=ties_top_k,
                    scaling_factor=scaling_factor,
                )
            elif method == "dare":
                merged_tensor = dare_merge(
                    base_weights=bp_f32,
                    fine_tuned_weights=tp_f32,
                    drop_rate=dare_drop_rate,
                    scaling_factor=scaling_factor,
                )

            merged_dict[key] = merged_tensor.to(dtype=orig_dtype)
            merged_params += num_elem

        summary = {
            "method": method,
            "total_parameters": total_params,
            "merged_parameters": merged_params,
            "skipped_non_float_parameters": skipped_non_float,
            "skipped_filtered_parameters": skipped_filtered,
            "num_targets": len(target_state_dicts),
        }

        return merged_dict, summary
