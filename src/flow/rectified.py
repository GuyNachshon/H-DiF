import torch
from torch import nn


def make_x0(cond, mode="noise", alpha=0.7):
    """Starting point x0 for the flow, matching cond's RGB shape (B,3,H,W).

    - "noise": pure Gaussian noise (best color/diversity, loose structure).
    - "tir": TIR channel broadcast to 3ch (best structure, color collapse).
    - "blend": alpha*noise + (1-alpha)*tir, interpolating the two.
    """
    tir3 = cond[:, 0:1].repeat(1, 3, 1, 1)
    if mode == "tir":
        return tir3
    noise = torch.randn_like(tir3)
    if mode == "noise":
        return noise
    if mode == "blend":
        return alpha * noise + (1 - alpha) * tir3
    raise ValueError(f"unknown x0_mode: {mode}")


def flow_batch(x0, x1, t):
    """x_t = (1-t)*x0 + t*x1 ; v = x1 - x0. t: [B] broadcast to [B,1,1,1]."""
    t = t.view(-1, 1, 1, 1)
    x_t = (1 - t) * x0 + t * x1
    v = x1 - x0
    return x_t, v


class RectifiedFlow(nn.Module):
    def __init__(self, model, t_sampling="uniform", cond_dropout=0.0):
        super().__init__()
        self.model = model
        self.t_sampling = t_sampling
        self.cond_dropout = cond_dropout

    def forward(self, x_t, t, cond, cache=None):
        x = torch.cat([x_t, cond], dim=1)
        # t is the flow time in [0,1]; the model reads it as sigma (log(sigma)/4 internally),
        # so clamp away from 0 to keep the log finite at the trajectory start.
        sigma = t.clamp_min(1e-4)
        return self.model(x, sigma, cache=cache)

    def loss(self, x0, x1, cond):
        if self.t_sampling == "logit_normal":
            t = torch.sigmoid(torch.randn(x0.shape[0], device=x0.device))
        else:
            t = torch.rand(x0.shape[0], device=x0.device)
        if self.cond_dropout > 0:
            keep = torch.rand(x0.shape[0], 1, 1, 1, device=x0.device) >= self.cond_dropout
            cond = torch.where(keep, cond, torch.zeros_like(cond))
        x_t, v = flow_batch(x0, x1, t)
        v_pred = self(x_t, t, cond)
        return torch.mean((v_pred - v) ** 2)


@torch.no_grad()
def trajectory_straightness(rf, x0, x1, cond, n_probes=8):
    """Variance of v_pred over t at fixed endpoints -> float; lower = straighter."""
    preds = []
    for i in range(n_probes):
        t = torch.full((x0.shape[0],), (i + 0.5) / n_probes, device=x0.device)
        x_t, _ = flow_batch(x0, x1, t)
        preds.append(rf(x_t, t, cond))
    preds = torch.stack(preds, dim=0)
    return preds.var(dim=0).mean().item()
