# Model Merging & Distillation Lab

A production-grade, tensor-level model merging framework implementing state-of-the-art parameter arithmetic algorithms: **SLERP** (Spherical Linear Interpolation), **TIES-Merging** (TRIm, Elect Sign, and Merge), and **DARE** (Drop And REscale).

---

## 🔬 Core Algorithms & Mathematics

### 1. Spherical Linear Interpolation (SLERP)
Linear interpolation ($\text{LERP}$) in high-dimensional weight spaces causes variance collapse toward the origin ($||\theta_{\text{LERP}}|| < ||\theta||$), degrading language model perplexity. SLERP interpolates along the great-circle arc on the unit hypersphere:

$$\theta(t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} \theta_0 + \frac{\sin(t\Omega)}{\sin\Omega} \theta_1$$

Where $\cos\Omega = \frac{\theta_0 \cdot \theta_1}{||\theta_0|| \, ||\theta_1||}$. When $\Omega < 10^{-5}$, it smoothly transitions to normalized linear interpolation.

- **Implementation**: [`merge/slerp.py`](merge/slerp.py) (`slerp`, `slerp_tensor`)
- **Key Characteristics**: Preserves geometric magnitude, prevents activation variance decay.

---

### 2. TIES-Merging (NeurIPS 2023)
Fine-tuned checkpoints suffer from **parameter interference** (redundant modifications and conflicting sign directions). TIES-Merging (Yadav et al.) eliminates interference in 3 steps:

1. **TRIm (Task Vector Sparsification)**:
   $$\tau_m = \theta_m - \theta_{\text{base}}, \quad \hat{\tau}_m = \text{TopK}_{k\%}(|\tau_m|)$$
2. **Elect Consensus Sign**:
   $$\gamma = \text{sign}\left(\sum_{m=1}^M \hat{\tau}_m\right)$$
3. **Disjoint Merge & Recombine**:
   Average only agreeing parameters ($\text{sign}(\hat{\tau}_{m, j}) = \gamma_j$) and add back to base:
   $$\theta_{\text{merged}} = \theta_{\text{base}} + \lambda \cdot \tau_{\text{TIES}}$$

- **Implementation**: [`merge/ties.py`](merge/ties.py) (`trim_task_vector`, `elect_signs`, `disjoint_merge`, `ties_merge`)
- **Key Characteristics**: Resolves gradient conflicts, combines models with complementary domain expertise.

---

### 3. DARE (Drop And REscale, 2024)
DARE (Yu et al.) eliminates up to 90% of fine-tuned delta parameters through Bernoulli masking while scaling remaining parameters by an unbiased expectation factor:

$$\tilde{\tau}_m = \frac{\tau_m \odot \text{Bernoulli}(1 - p)}{1 - p}$$

Because $\mathbb{E}[\tilde{\tau}_m] = \tau_m$, models can be merged without destructive interference:
$$\theta_{\text{merged}} = \theta_{\text{base}} + \frac{\lambda}{M} \sum_{m=1}^M \tilde{\tau}_m$$

- **Implementation**: [`merge/dare.py`](merge/dare.py) (`dare_drop_and_rescale`, `dare_merge`)
- **Key Characteristics**: Extreme sparsity with minimal task degradation, prevents parameter explosion.

---

## 🚀 ModelMerger Pipeline

[`merge/pipeline.py`](merge/pipeline.py) provides a unified interface for merging full PyTorch state dictionaries (from HuggingFace checkpoints or custom modules):

```python
from merge.pipeline import ModelMerger
import torch

merger = ModelMerger(device="cpu")

# Merge base with fine-tuned models
merged_state_dict, summary = merger.merge_state_dicts(
    method="ties",
    base_state_dict=base_weights,
    target_state_dicts=[coding_expert_weights, math_expert_weights],
    ties_top_k=0.2,       # Retain top 20% magnitude
    scaling_factor=1.0,   # Task vector multiplier
    filter_regex=r"layers\.\d+\.(self_attn|mlp)", # Layer filtering
)

print(summary)
# {
#   "method": "ties",
#   "total_parameters": 494032896,
#   "merged_parameters": 450123000,
#   "skipped_non_float_parameters": 4096,
#   "skipped_filtered_parameters": 43905800,
#   "num_targets": 2
# }
```

### Safety & Engineering Features
- **Buffer Preservation**: Automatic identification and passthrough of non-floating point buffers (e.g. `position_ids`, `step_counters`).
- **Precision Guard**: Automatic float32 casting during trigonometry / vector summation to prevent fp16 underflow/overflow.
- **Layer Regex Filter**: Selective merging of specific transformer blocks (e.g., merging attention layers while freezing MLPs).
- **Key Verification**: Strict architecture key matching with clear diagnostics.

---

## 🧪 Automated Test Suite (34/34 Passing)

Run the comprehensive pytest suite:
```bash
python -m pytest tests/ -v
```

| Test File | Focus | Test Count |
|---|---|---|
| [`tests/test_merge.py`](tests/test_merge.py) | SLERP 1D and tensor interpolation | 3 passed |
| [`tests/test_ties.py`](tests/test_ties.py) | Trimming, Sign Election, Disjoint Merge, Conflict Resolution | 11 passed |
| [`tests/test_dare.py`](tests/test_dare.py) | Unbiased expectation, Bernoulli sparsity, multi-model DARE | 7 passed |
| [`tests/test_pipeline.py`](tests/test_pipeline.py) | End-to-end `ModelMerger` state dict operations, layer filtering | 6 passed |
| [`tests/test_edge_cases.py`](tests/test_edge_cases.py) | Zero norms, mismatched shapes, single-element tensors, exceptions | 7 passed |
| **Total** | | **34 passed** |
