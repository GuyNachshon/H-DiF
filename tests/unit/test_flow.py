import math

import torch

from flow.rectified import RectifiedFlow, flow_batch, trajectory_straightness
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


class _IdentityModel(torch.nn.Module):
    def forward(self, x, sigma, cache=None):
        return x[:, :3]  # v_pred = x_t channels, ignores cond channels


def test_loss_logit_normal_t_sampling_runs():
    rf = RectifiedFlow(_IdentityModel(), t_sampling="logit_normal")
    x0, x1, cond = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8), torch.randn(4, 2, 8, 8)
    loss = rf.loss(x0, x1, cond)
    assert math.isfinite(loss.item())


def test_loss_cond_dropout_zeroes_some_samples():
    torch.manual_seed(0)
    rf = RectifiedFlow(_IdentityModel(), cond_dropout=1.0)
    x0, x1 = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8)
    cond = torch.ones(4, 2, 8, 8)
    seen = {}
    orig_forward = rf.forward

    def spy_forward(x_t, t, cond, cache=None):
        seen["cond"] = cond
        return orig_forward(x_t, t, cond, cache)

    rf.forward = spy_forward
    rf.loss(x0, x1, cond)
    assert torch.all(seen["cond"] == 0)  # cond_dropout=1.0 -> always dropped
