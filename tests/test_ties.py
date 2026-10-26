import pytest
import torch
from merge.ties import trim_task_vector, elect_signs, disjoint_merge, ties_merge


def test_trim_task_vector_retention_ratio():
    # Tensor with 10 elements: 1 to 10
    tv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    trimmed = trim_task_vector(tv, top_k_fraction=0.3)
    
    # Exactly top 30% (3 items: 8.0, 9.0, 10.0) must be non-zero
    non_zero_mask = (trimmed != 0)
    assert non_zero_mask.sum().item() == 3
    assert trimmed[9] == 10.0
    assert trimmed[8] == 9.0
    assert trimmed[7] == 8.0
    assert trimmed[0] == 0.0


def test_trim_task_vector_negative_magnitudes():
    # Negative values with large magnitude should be retained
    tv = torch.tensor([-15.0, 2.0, 0.5, -0.1, 10.0])
    trimmed = trim_task_vector(tv, top_k_fraction=0.4)
    # Top 2 by absolute value: -15.0 and 10.0
    assert trimmed[0] == -15.0
    assert trimmed[4] == 10.0
    assert trimmed[1] == 0.0
    assert trimmed[2] == 0.0
    assert trimmed[3] == 0.0


def test_trim_task_vector_full_retention():
    tv = torch.randn(4, 4)
    trimmed = trim_task_vector(tv, top_k_fraction=1.0)
    assert torch.equal(tv, trimmed)


def test_trim_task_vector_multidimensional():
    tv = torch.randn(2, 3, 4)
    trimmed = trim_task_vector(tv, top_k_fraction=0.5)
    assert (trimmed != 0).sum().item() == 12


def test_elect_signs_unanimous():
    v1 = torch.tensor([1.0, -2.0, 3.0])
    v2 = torch.tensor([2.0, -1.0, 4.0])
    signs = elect_signs([v1, v2])
    assert torch.equal(signs, torch.tensor([1.0, -1.0, 1.0]))


def test_elect_signs_majority_and_magnitude():
    # Sum determines direction:
    # index 0: 5.0 + (-2.0) = +3.0 -> +1.0
    # index 1: -10.0 + 3.0 = -7.0 -> -1.0
    # index 2: 4.0 + (-4.0) = 0.0 -> 0.0
    v1 = torch.tensor([5.0, -10.0, 4.0])
    v2 = torch.tensor([-2.0, 3.0, -4.0])
    signs = elect_signs([v1, v2])
    assert torch.equal(signs, torch.tensor([1.0, -1.0, 0.0]))


def test_disjoint_merge_agreement():
    # Elected signs: [+1, -1]
    # v1: [+2, -4] (both agree)
    # v2: [+4, +6] (first agrees, second disagrees!)
    v1 = torch.tensor([2.0, -4.0])
    v2 = torch.tensor([4.0, 6.0])
    elected = torch.tensor([1.0, -1.0])
    
    merged = disjoint_merge([v1, v2], elected)
    
    # At index 0: both agree -> mean(2.0, 4.0) = 3.0
    # At index 1: only v1 agrees (-4.0), v2 disagrees (+6.0) -> mean(-4.0) = -4.0
    assert merged[0].item() == pytest.approx(3.0)
    assert merged[1].item() == pytest.approx(-4.0)


def test_disjoint_merge_all_disagree():
    # If elected sign is 0 or all disagree, result is 0
    v1 = torch.tensor([2.0, -4.0])
    v2 = torch.tensor([-2.0, 4.0])
    elected = torch.tensor([0.0, 0.0])
    merged = disjoint_merge([v1, v2], elected)
    assert torch.equal(merged, torch.zeros(2))


def test_ties_merge_end_to_end_single_model():
    base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ft = torch.tensor([[2.0, 2.0], [3.0, 8.0]])
    # Task vector: [[1.0, 0.0], [0.0, 4.0]]
    # top_k=0.5 retains top 2 items (4.0 and 1.0)
    merged = ties_merge(base, [ft], top_k_fraction=0.5, scaling_factor=1.0)
    assert torch.isclose(merged, ft).all()


def test_ties_merge_two_models_resolving_interference():
    base = torch.zeros(4)
    # Model A specializes in indices 0 and 1
    model_a = torch.tensor([10.0, -5.0, 0.1, 0.0])
    # Model B specializes in indices 1 and 2, but has opposing gradient on index 1
    model_b = torch.tensor([0.0, 1.0, 8.0, 0.0])
    
    # On index 1, Model A has -5.0, Model B has +1.0. Sum is -4.0 -> elected sign is -1.
    # Model B's +1.0 will be discarded as conflicting!
    # On index 2, Model A has +0.1 and Model B has +8.0. Both agree with +1 sign -> mean is (0.1 + 8.0) / 2 = 4.05
    merged = ties_merge(base, [model_a, model_b], top_k_fraction=0.75, scaling_factor=1.0)
    
    assert merged[0].item() == pytest.approx(10.0) # From model A only (model B is 0.0)
    assert merged[1].item() == pytest.approx(-5.0) # Conflicting direction filtered out; only model A kept
    assert merged[2].item() == pytest.approx(4.05) # Both agree on sign (+), averaged correctly
    assert merged[3].item() == pytest.approx(0.0)


def test_ties_merge_scaling_factor():
    base = torch.ones(2)
    ft1 = torch.tensor([3.0, 5.0])
    # Task vector: [2.0, 4.0]
    merged_half = ties_merge(base, [ft1], top_k_fraction=1.0, scaling_factor=0.5)
    # base + 0.5 * [2, 4] = [1 + 1, 1 + 2] = [2, 3]
    assert torch.allclose(merged_half, torch.tensor([2.0, 3.0]))
