import torch

from models.build import build_hdit
from temporal.cache import AttnResidualCache


def tiny_model():
    return build_hdit(
        {
            "in_channels": 5,
            "out_channels": 3,
            "patch_size": [4, 4],
            "widths": [32, 64],
            "depths": [1, 1],
            "d_ffs": [64, 128],
            "self_attns": [
                {"type": "shifted-window", "d_head": 32, "window_size": 8},
                {"type": "global", "d_head": 32},
            ],
            "dropout": [0.0, 0.0],
            "mapping": {"depth": 1, "width": 64, "d_ff": 128, "dropout": 0.0},
            "size": 32,
        }
    )


def test_cache_stores_detached():
    model = tiny_model()
    cache = AttnResidualCache()
    x = torch.randn(2, 5, 32, 32)
    sigma = torch.rand(2)
    model(x, sigma, cache=cache)
    model(x, sigma, cache=cache)
    assert cache._store
    # bottleneck: side = 32/4/2 = 4 -> hw = 16, heads = width/d_head = 64/32 = 2
    for t in cache._store.values():
        assert t.requires_grad is False
        assert t.shape == (2, 2, 16, 16)


def test_cache_changes_output():
    model = tiny_model()
    # patch_out and the attention out_proj are zero-initialized, which would make every
    # output identically zero and mask the bias path — randomize them so signal propagates.
    with torch.no_grad():
        for p in model.parameters():
            if torch.count_nonzero(p) == 0:
                p.normal_(0, 0.02)
    x = torch.randn(2, 5, 32, 32)
    sigma = torch.rand(2)
    cache = AttnResidualCache()
    model(x, sigma, cache=cache)  # populate t-1
    with_cache = model(x, sigma, cache=cache)
    no_cache = model(x, sigma, cache=None)
    assert not torch.allclose(with_cache, no_cache)


def test_reset_clears():
    model = tiny_model()
    cache = AttnResidualCache()
    x = torch.randn(2, 5, 32, 32)
    sigma = torch.rand(2)
    model(x, sigma, cache=cache)
    block = next(iter(cache._store))
    cache.reset()
    assert not cache._store

    class B:
        pass

    assert cache.bias(B(), (2, 2, 16, 16)) == 0.0
