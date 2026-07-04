import torch

from models.build import build_hdit
from models.hdit import GlobalTransformerLayer


def tiny_cfg():
    return {
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


def test_forward_shape():
    model = build_hdit(tiny_cfg())
    x = torch.randn(2, 5, 32, 32)
    sigma = torch.rand(2)
    out = model(x, sigma)
    assert out.shape == (2, 3, 32, 32)
    assert out.dtype == torch.float32


def test_use_cache_flags():
    model = build_hdit(tiny_cfg())
    for layer in model.mid_level:
        if isinstance(layer, GlobalTransformerLayer):
            assert layer.self_attn.use_cache is True
    outer_global = [
        l for level in model.down_levels for l in level if isinstance(l, GlobalTransformerLayer)
    ]
    assert all(l.self_attn.use_cache is False for l in outer_global)
