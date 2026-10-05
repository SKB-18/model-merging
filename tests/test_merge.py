import pytest
import math
from merge.slerp import slerp

def test_slerp_basic():
    v0 = [1.0, 0.0]
    v1 = [0.0, 1.0]
    
    # At t=0, it should be v0
    res_0 = slerp(0.0, v0, v1)
    assert math.isclose(res_0[0], 1.0)
    assert math.isclose(res_0[1], 0.0)
    
    # At t=1, it should be v1
    res_1 = slerp(1.0, v0, v1)
    assert math.isclose(res_1[0], 0.0, abs_tol=1e-9)
    assert math.isclose(res_1[1], 1.0)
    
    # At t=0.5, it should be exactly in the middle of the quarter circle
    res_half = slerp(0.5, v0, v1)
    # sin(45 deg) = sqrt(2)/2 approx 0.707
    expected = math.sqrt(2)/2
    assert math.isclose(res_half[0], expected, abs_tol=1e-5)
    assert math.isclose(res_half[1], expected, abs_tol=1e-5)

def test_slerp_collinear():
    v0 = [1.0, 1.0]
    v1 = [2.0, 2.0]
    
    # Linear fallback should trigger
    res_half = slerp(0.5, v0, v1)
    assert math.isclose(res_half[0], 1.5)
    assert math.isclose(res_half[1], 1.5)
