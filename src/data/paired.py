import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)  # avoid per-worker thread pool contention under DataLoader workers


def _resize_center_crop(img, size):
    h, w = img.shape[:2]
    scale = size / min(h, w)
    img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2
    return img[top : top + size, left : left + size]


class PairedThermalRGB(Dataset):
    """root/{split}/clips/<clip_id>/frame_%04d.tir.png + .rgb.png

    Returns {'tir': [2,H,W] float32 (normalized TIR + cv2.Canny/255), 'rgb': [3,H,W] in [-1,1],
             'clip_id': str, 'frame_idx': int}
    """

    def __init__(self, root, split, size=256, clip_len=1):
        self.size = size
        self.clip_len = clip_len
        clips_root = os.path.join(root, split, "clips")
        if not os.path.isdir(clips_root):
            raise FileNotFoundError(f"data root missing: {clips_root}")
        self.frames = []
        for clip_dir in sorted(glob.glob(os.path.join(clips_root, "*"))):
            clip_id = os.path.basename(clip_dir)
            tirs = sorted(glob.glob(os.path.join(clip_dir, "*.tir.png")))
            for f in tirs:
                idx = int(os.path.basename(f).split(".")[0].split("_")[-1])
                self.frames.append((clip_id, clip_dir, idx))
        if not self.frames:
            raise FileNotFoundError(f"no frames found under {clips_root}")

    def __len__(self):
        return len(self.frames)

    def _load_one(self, clip_dir, idx):
        tir = cv2.imread(os.path.join(clip_dir, f"frame_{idx:04d}.tir.png"), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(os.path.join(clip_dir, f"frame_{idx:04d}.rgb.png"), cv2.IMREAD_COLOR)
        tir = _resize_center_crop(tir, self.size)
        rgb = _resize_center_crop(rgb, self.size)
        canny = cv2.Canny(tir, 100, 200)
        tir_norm = tir.astype(np.float32) / 255.0
        cond = np.stack([tir_norm, canny.astype(np.float32) / 255.0], axis=0)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        rgb = np.transpose(rgb, (2, 0, 1))
        return torch.from_numpy(cond), torch.from_numpy(rgb)

    def __getitem__(self, i):
        clip_id, clip_dir, idx = self.frames[i]
        if self.clip_len > 1:
            conds, rgbs = [], []
            for j in range(self.clip_len):
                c, r = self._load_one(clip_dir, idx + j)
                conds.append(c)
                rgbs.append(r)
            return {"tir": torch.stack(conds), "rgb": torch.stack(rgbs), "clip_id": clip_id, "frame_idx": idx}
        cond, rgb = self._load_one(clip_dir, idx)
        return {"tir": cond, "rgb": rgb, "clip_id": clip_id, "frame_idx": idx}
