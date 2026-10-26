import pytest
import math
import torch
from merge.slerp import slerp, slerp_tensor


def test_slerp_basic():
    v0 = [1.0, 0.0]
    v1 = [0.0, 1.0]
    
    res_0 = slerp(0.0, v0, v1)
    assert math.isclose(res_0[0], 1.0)
    assert math.isclose(res_0[1], 0.0)
    
    res_1 = slerp(1.0, v0, v1)
    assert math.isclose(res_1[0], 0.0, abs_tol=1e-9)
    assert math.isclose(res_1[1], 1.0)
    
    res_half = slerp(0.5, v0, v1)
    expected = math.sqrt(2)/2
    assert math.isclose(res_half[0], expected, abs_tol=1e-5)
    assert math.isclose(res_half[1], expected, abs_tol=1e-5)


def test_slerp_collinear():
    v0 = [1.0, 1.0]
    v1 = [2.0, 2.0]
    res_half = slerp(0.5, v0, v1)
    assert math.isclose(res_half[0], 1.5)
    assert math.isclose(res_half[1], 1.5)


def test_slerp_tensor():
    t0 = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    t1 = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    
    # Midpoint
    res = slerp_tensor(0.5, t0, t1)
    assert res.shape == (2, 2)
    expected = math.sqrt(2)/2
    assert torch.isclose(res[0, 0], torch.tensor(expected), atol=1e-4)
    assert torch.isclose(res[0, 1], torch.tensor(expected), atol=1e-4)
