import torch


@torch.no_grad()
def euler(rf, x0, cond, steps, cache=None):
    """Integrate t: 0->1 with uniform steps. NFE = steps."""
    x = x0
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        v = rf(x, t, cond, cache=cache)
        x = x + dt * v
    return x


@torch.no_grad()
def midpoint(rf, x0, cond, steps, cache=None):
    """Integrate t: 0->1 with uniform steps. NFE = 2*steps."""
    x = x0
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        v1 = rf(x, t, cond, cache=cache)
        t_mid = torch.full((x.shape[0],), (i + 0.5) * dt, device=x.device)
        v2 = rf(x + 0.5 * dt * v1, t_mid, cond, cache=cache)
        x = x + dt * v2
    return x
