import numpy as np

from train import _colorfulness, _edge_recall


def test_edge_recall_perfect_match():
    tir = np.zeros((32, 32), dtype=np.float32)
    tir[:, 16:] = 1.0  # vertical edge at column 16
    pred = np.full((32, 32, 3), -1.0, dtype=np.float32)
    pred[:, 16:] = 1.0  # same edge, gray -> RGB
    assert _edge_recall(tir, pred) > 0.9


def test_edge_recall_flat_pred_is_near_zero():
    tir = np.zeros((32, 32), dtype=np.float32)
    tir[:, 16:] = 1.0
    pred = np.zeros((32, 32, 3), dtype=np.float32)  # flat gray, no edges
    assert _edge_recall(tir, pred) < 0.1


def test_colorfulness_gray_is_near_zero():
    gray = np.full((16, 16, 3), 0.5, dtype=np.float32)
    assert _colorfulness(gray) < 1e-6


def test_colorfulness_random_exceeds_gray():
    rng = np.random.default_rng(0)
    gray = np.full((16, 16, 3), 0.5, dtype=np.float32)
    colorful = rng.random((16, 16, 3), dtype=np.float32)
    assert _colorfulness(colorful) > _colorfulness(gray)
