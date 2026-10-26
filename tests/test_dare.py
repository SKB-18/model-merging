import pytest
import torch
from merge.dare import dare_drop_and_rescale, dare_merge


def test_dare_drop_rate_zero():
    tv = torch.randn(10, 10)
    rescaled = dare_drop_and_rescale(tv, drop_rate=0.0)
    assert torch.equal(tv, rescaled)


def test_dare_seed_reproducibility():
    tv = torch.randn(50, 50)
    out1 = dare_drop_and_rescale(tv, drop_rate=0.5, seed=42)
    out2 = dare_drop_and_rescale(tv, drop_rate=0.5, seed=42)
    assert torch.equal(out1, out2)


def test_dare_drop_rate_sparsity():
    torch.manual_seed(123)
    tv = torch.ones(10000)
    drop_rate = 0.7
    out = dare_drop_and_rescale(tv, drop_rate=drop_rate, seed=123)
    
    # Check fraction of zeros is approximately 0.7
    zeros = (out == 0.0).sum().item()
    zero_ratio = zeros / 10000.0
    assert 0.67 < zero_ratio < 0.73


def test_dare_expectation_unbiased():
    # Mathematical property: E[DARE(v)] = v
    # Surviving elements are rescaled by 1 / (1 - drop_rate)
    tv = torch.full((50000,), 4.0)
    drop_rate = 0.6
    out = dare_drop_and_rescale(tv, drop_rate=drop_rate, seed=999)
    
    # Expected mean should be close to 4.0
    empirical_mean = out.mean().item()
    assert abs(empirical_mean - 4.0) < 0.1


def test_dare_merge_single_model_no_drop():
    base = torch.tensor([1.0, 2.0, 3.0])
    ft = torch.tensor([4.0, 5.0, 6.0])
    merged = dare_merge(base, [ft], drop_rate=0.0, scaling_factor=1.0)
    assert torch.allclose(merged, ft)


def test_dare_merge_multi_model_scaling():
    base = torch.zeros(4)
    ft1 = torch.tensor([2.0, 2.0, 2.0, 2.0])
    ft2 = torch.tensor([4.0, 4.0, 4.0, 4.0])
    
    # Without drop: avg task vector is [3.0, 3.0, 3.0, 3.0]
    # With scaling factor 0.5: merged = [1.5, 1.5, 1.5, 1.5]
    merged = dare_merge(base, [ft1, ft2], drop_rate=0.0, scaling_factor=0.5)
    assert torch.allclose(merged, torch.tensor([1.5, 1.5, 1.5, 1.5]))


def test_dare_invalid_drop_rate():
    tv = torch.ones(5)
    with pytest.raises(AssertionError):
        dare_drop_and_rescale(tv, drop_rate=-0.1)
    with pytest.raises(AssertionError):
        dare_drop_and_rescale(tv, drop_rate=1.0)
