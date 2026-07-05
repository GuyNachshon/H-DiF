import math

import torch

from flow.rectified import RectifiedFlow, _grad_mag, flow_batch, make_x0, trajectory_straightness
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


def test_make_x0_shapes():
    cond = torch.randn(4, 2, 8, 8)
    for mode in ("noise", "tir", "blend"):
        x0 = make_x0(cond, mode=mode)
        assert x0.shape == (4, 3, 8, 8)


def test_make_x0_tir_alpha_zero_is_exact_broadcast():
    cond = torch.randn(4, 2, 8, 8)
    tir3 = cond[:, 0:1].repeat(1, 3, 1, 1)
    assert torch.allclose(make_x0(cond, mode="blend", alpha=0.0), tir3)
    assert torch.allclose(make_x0(cond, mode="tir"), tir3)


def test_make_x0_blend_alpha_one_is_noise_like():
    torch.manual_seed(0)
    cond = torch.randn(64, 2, 16, 16)
    x0 = make_x0(cond, mode="blend", alpha=1.0)
    assert abs(x0.std().item() - 1.0) < 0.05


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


def test_grad_mag_peaks_at_stripe_boundary():
    img = -torch.ones(1, 3, 16, 16)
    img[:, :, :, 8:] = 1.0  # vertical stripe boundary at column 8
    mag = _grad_mag(img)
    peak_col = mag[0, 0, 8, :].argmax().item()
    assert peak_col in (7, 8)


class _V0Model(torch.nn.Module):
    """v_pred = 0 everywhere, regardless of edge_weight."""

    def forward(self, x, sigma, cache=None):
        return torch.zeros_like(x[:, :3])


def test_edge_loss_adds_nonnegative_mass():
    torch.manual_seed(0)
    x0, x1 = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8)
    cond = torch.rand(4, 2, 8, 8)

    torch.manual_seed(1)
    rf0 = RectifiedFlow(_V0Model(), edge_weight=0.0)
    loss0 = rf0.loss(x0, x1, cond)

    torch.manual_seed(1)
    rf1 = RectifiedFlow(_V0Model(), edge_weight=0.5)
    loss1 = rf1.loss(x0, x1, cond)

    assert math.isfinite(loss0.item()) and math.isfinite(loss1.item())
    assert loss1.item() > loss0.item()


def test_x1_hat_round_trip_with_exact_v():
    torch.manual_seed(0)
    x0, x1 = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8)
    t = torch.rand(4)
    x_t, v = flow_batch(x0, x1, t)
    x1_hat = x_t + (1 - t.view(-1, 1, 1, 1)) * v
    assert torch.allclose(x1_hat, x1, atol=1e-6)
