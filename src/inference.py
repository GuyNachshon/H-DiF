import argparse

import cv2
import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from data.paired import _resize_center_crop
from flow.rectified import RectifiedFlow
from models.build import build_hdit
from sampling.ode import midpoint
from temporal.cache import AttnResidualCache


def _cond_from_frame(frame, size):
    tir = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tir = _resize_center_crop(tir, size)
    canny = cv2.Canny(tir, 100, 200)
    tir_norm = tir.astype(np.float32) / 255.0
    cond = np.stack([tir_norm, canny.astype(np.float32) / 255.0], axis=0)
    return torch.from_numpy(cond)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", default="config/rectified_flow.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    size = cfg["data"]["size"]
    steps = args.steps if args.steps is not None else cfg["sampling"]["steps"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_hdit(dict(cfg["model"], size=size)).to(device)
    model.load_state_dict(load_file(args.checkpoint))
    model.eval()
    rf = RectifiedFlow(model).to(device)

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))

    cache = AttnResidualCache(gamma=cfg["temporal"]["gamma"])
    cache.reset()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cond = _cond_from_frame(frame, size).to(device)
        x0 = cond[:, 0:1].repeat(1, 3, 1, 1)
        out = midpoint(rf, x0, cond, steps, cache=cache)
        img = ((out[0].clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy().transpose(1, 2, 0)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
