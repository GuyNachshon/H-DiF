import math

import torch

from flow.rectified import flow_batch, trajectory_straightness
from sampling.ode import euler, midpoint


def test_flow_batch_boundaries():
    x0 = torch.randn(3, 3, 8, 8)
    x1 = torch.randn(3, 3, 8, 8)
    xt0, _ = flow_batch(x0, x1, torch.zeros(3))
    xt1, _ = flow_batch(x0, x1, torch.ones(3))
    assert torch.allclose(xt0, x0)
    assert torch.allclose(xt1, x1)
    assert xt0.shape == x0.shape


class StubRF:
    def __init__(self):
        self.calls = 0

    def __call__(self, x, t, cond, cache=None):
        self.calls += 1
        return torch.zeros_like(x)


def test_euler_nfe():
    rf = StubRF()
    euler(rf, torch.zeros(2, 3, 8, 8), None, steps=4)
    assert rf.calls == 4


def test_midpoint_nfe():
    rf = StubRF()
    midpoint(rf, torch.zeros(2, 3, 8, 8), None, steps=2)
    assert rf.calls == 4


def test_straightness_finite():
    rf = StubRF()
    s = trajectory_straightness(rf, torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8), None)
    assert math.isfinite(s) and s >= 0
