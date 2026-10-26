import pytest
import torch
from merge.slerp import slerp, slerp_tensor
from merge.ties import trim_task_vector, elect_signs, disjoint_merge, ties_merge
from merge.dare import dare_drop_and_rescale, dare_merge
from merge.pipeline import ModelMerger


def test_slerp_zero_vectors():
    v0 = [0.0, 0.0, 0.0]
    v1 = [1.0, 2.0, 3.0]
    res = slerp(0.5, v0, v1)
    assert res == [0.0, 0.0, 0.0]


def test_slerp_tensor_zero_vectors():
    t0 = torch.zeros(3, 3)
    t1 = torch.ones(3, 3)
    # Norm of t0 is 0, should fallback gracefully to lerp
    res = slerp_tensor(0.5, t0, t1)
    assert torch.allclose(res, torch.full((3, 3), 0.5))


def test_slerp_tensor_shape_mismatch():
    t0 = torch.ones(2, 2)
    t1 = torch.ones(3, 3)
    with pytest.raises(AssertionError):
        slerp_tensor(0.5, t0, t1)


def test_ties_single_element_tensor():
    base = torch.tensor([1.0])
    ft = torch.tensor([3.0])
    merged = ties_merge(base, [ft], top_k_fraction=1.0)
    assert merged.item() == 3.0


def test_ties_all_zeros():
    base = torch.zeros(5)
    ft1 = torch.zeros(5)
    ft2 = torch.zeros(5)
    merged = ties_merge(base, [ft1, ft2], top_k_fraction=0.5)
    assert torch.equal(merged, torch.zeros(5))


def test_dare_near_one_drop_rate():
    tv = torch.ones(100)
    out = dare_drop_and_rescale(tv, drop_rate=0.99, seed=1)
    # Most will be 0, non-zeros will be rescaled to ~100
    assert (out == 0).sum() > 80


def test_pipeline_empty_target_raises():
    merger = ModelMerger()
    base = {"weight": torch.ones(2)}
    with pytest.raises(ValueError):
        merger.merge_state_dicts(
            method="ties",
            base_state_dict=base,
            target_state_dicts=[],
        )
