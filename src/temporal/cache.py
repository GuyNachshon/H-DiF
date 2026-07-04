import torch


def throttle_gamma(gamma_base: float, flow_magnitude: float, k: float = 4.0) -> float:
    """gamma_base / (1 + k * flow_magnitude) — monotonically non-increasing in flow magnitude."""
    return gamma_base / (1.0 + k * flow_magnitude)


class AttnResidualCache:
    """Detached attention matrices from frame t-1, keyed by block. Reset between clips."""

    def __init__(self, gamma=0.35):
        self._store = {}
        self.gamma = gamma
        self._gamma_eff = gamma

    def bias(self, block, q_shape):
        prev = self._store.get(id(block))
        if prev is None or prev.shape != (q_shape[0], q_shape[1], q_shape[2], q_shape[2]):
            return 0.0
        return self._gamma_eff * prev

    def store(self, block, attn):
        self._store[id(block)] = attn.detach()

    def reset(self):
        self._store.clear()

    def throttle(self, flow_mag):
        self._gamma_eff = throttle_gamma(self.gamma, flow_mag)


def _demo():
    c = AttnResidualCache(gamma=0.4)

    class B:
        pass

    b = B()
    assert c.bias(b, (2, 3, 5, 5)) == 0.0
    a = torch.randn(2, 3, 5, 5, requires_grad=True)
    c.store(b, a)
    assert c._store[id(b)].requires_grad is False
    bias = c.bias(b, (2, 3, 5, 5))
    assert torch.allclose(bias, 0.4 * a.detach())
    assert c.bias(b, (2, 3, 7, 7)) == 0.0  # shape mismatch -> scalar 0
    c.throttle(1.0)
    assert c._gamma_eff < c.gamma
    c.reset()
    assert c.bias(b, (2, 3, 5, 5)) == 0.0
    # monotonicity
    mags = [0, 0.5, 1, 2, 10]
    vals = [throttle_gamma(0.35, m) for m in mags]
    assert vals[0] == 0.35
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    print("cache demo ok")


if __name__ == "__main__":
    _demo()
