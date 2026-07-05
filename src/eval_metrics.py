"""Val-time image metrics shared by src/train.py and src/sd/train_controlnet.py."""

import cv2
import numpy as np

_CANNY_THRESH = (100, 200)  # matches data/paired.py and inference.py
_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def edge_ssim(tir_np, pred_np):
    """Canny-edge SSIM between input TIR channel and predicted RGB (Phase-1 gate, PLAN.md 1.4)."""
    from skimage.metrics import structural_similarity

    tir_u8 = (tir_np * 255.0).clip(0, 255).astype("uint8")
    pred_gray = cv2.cvtColor(((pred_np + 1.0) * 127.5).clip(0, 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
    edges_tir = cv2.Canny(tir_u8, *_CANNY_THRESH)
    edges_pred = cv2.Canny(pred_gray, *_CANNY_THRESH)
    return structural_similarity(edges_tir, edges_pred, data_range=255)


def edge_recall(tir_np, pred_np):
    """Fraction of TIR Canny-edge pixels covered by a 1px-dilated pred-edge mask."""
    tir_u8 = (tir_np * 255.0).clip(0, 255).astype("uint8")
    pred_gray = cv2.cvtColor(((pred_np + 1.0) * 127.5).clip(0, 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
    edges_tir = cv2.Canny(tir_u8, *_CANNY_THRESH) > 0
    edges_pred = cv2.Canny(pred_gray, *_CANNY_THRESH)
    edges_pred_dilated = cv2.dilate(edges_pred, _DILATE_KERNEL) > 0
    n_tir_edges = edges_tir.sum()
    if n_tir_edges == 0:
        return 1.0  # no edges to recall -> vacuously covered
    return (edges_tir & edges_pred_dilated).sum() / n_tir_edges


def colorfulness(img_01):
    """Hasler-Susstrunk colorfulness metric. img_01: [H,W,3] in [0,1]."""
    r, g, b = img_01[..., 0], img_01[..., 1], img_01[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
