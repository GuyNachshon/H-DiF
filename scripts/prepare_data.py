"""Convert raw KAIST / LLVIP downloads into the loader layout:

    <out>/{train,val}/clips/<clip_id>/frame_%04d.tir.png + frame_%04d.rgb.png

See src/data/paired.py for the consumer.
"""

import argparse
import glob
import os
import re

import cv2


def _write_pair(out_clip_dir, idx, tir_path, rgb_path):
    tir = cv2.imread(tir_path, cv2.IMREAD_GRAYSCALE)
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if tir is None or rgb is None:
        return False
    os.makedirs(out_clip_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_clip_dir, f"frame_{idx:04d}.tir.png"), tir)
    cv2.imwrite(os.path.join(out_clip_dir, f"frame_{idx:04d}.rgb.png"), rgb)
    return True


def _prep_kaist_train(src, out, stride, counts):
    # kaist_train/setXX/VYYY/{lwir,visible}/I*.jpg — dense video, sets 00-05, needs subsampling.
    skipped = 0
    total_seen = 0
    for vdir in sorted(glob.glob(os.path.join(src, "kaist_train", "set*", "V*"))):
        set_name, v_name = os.path.basename(os.path.dirname(vdir)), os.path.basename(vdir)
        lwir_files = sorted(glob.glob(os.path.join(vdir, "lwir", "I*.jpg")))
        clip_id = f"kaist_{set_name}_{v_name}"
        out_idx = 0
        for lwir_path in lwir_files[::stride]:
            rgb_path = os.path.join(vdir, "visible", os.path.basename(lwir_path))
            out_clip_dir = os.path.join(out, "train", "clips", clip_id)
            if _write_pair(out_clip_dir, out_idx, lwir_path, rgb_path):
                counts["train"] += 1
                out_idx += 1
            else:
                skipped += 1
            total_seen += 1
            if total_seen % 1000 == 0:
                print(f"...{total_seen} frames processed")
    return skipped


_TEST_NAME_RE = re.compile(r"(set\d+)_(V\d+)_I(\d+)_(?:lwir|visible)")


def _prep_kaist_test(src, out, counts):
    # kaist_test/kaist_test_{lwir,visible}/setXX_VYYY_IXXXXX_{lwir,visible}.png
    # Flattened, already subsampled — group by (set, V), renumber contiguously, no stride.
    skipped = 0
    total_seen = 0
    lwir_dir = os.path.join(src, "kaist_test", "kaist_test_lwir")
    vis_dir = os.path.join(src, "kaist_test", "kaist_test_visible")
    clips = {}  # (set_name, v_name) -> [(idx, lwir_path, rgb_path), ...]
    for lwir_path in sorted(glob.glob(os.path.join(lwir_dir, "*_lwir.png"))):
        m = _TEST_NAME_RE.match(os.path.basename(lwir_path))
        if not m:
            continue
        set_name, v_name, frame_idx = m.groups()
        fname = f"{set_name}_{v_name}_I{frame_idx}_visible.png"
        rgb_path = os.path.join(vis_dir, fname)
        clips.setdefault((set_name, v_name), []).append((int(frame_idx), lwir_path, rgb_path))

    for (set_name, v_name), frames in clips.items():
        clip_id = f"kaist_{set_name}_{v_name}"
        out_clip_dir = os.path.join(out, "val", "clips", clip_id)
        for out_idx, (_, lwir_path, rgb_path) in enumerate(sorted(frames)):
            if _write_pair(out_clip_dir, out_idx, lwir_path, rgb_path):
                counts["val"] += 1
            else:
                skipped += 1
            total_seen += 1
            if total_seen % 1000 == 0:
                print(f"...{total_seen} frames processed")
    return skipped


def prep_kaist(src, out, stride=5):
    counts = {"train": 0, "val": 0}
    skipped = _prep_kaist_train(src, out, stride, counts)
    skipped += _prep_kaist_test(src, out, counts)
    print(f"kaist: skipped {skipped} unreadable/missing pairs")
    print(f"kaist: train={counts['train']} val={counts['val']}")


def prep_llvip(src, out):
    counts = {"train": 0, "val": 0}
    skipped = 0
    total_seen = 0
    for raw_split, split in [("train", "train"), ("test", "val")]:
        for tir_path in sorted(glob.glob(os.path.join(src, "infrared", raw_split, "*.jpg"))):
            basename = os.path.splitext(os.path.basename(tir_path))[0]
            rgb_path = os.path.join(src, "visible", raw_split, f"{basename}.jpg")
            clip_id = f"llvip_{basename}"
            out_clip_dir = os.path.join(out, split, "clips", clip_id)
            if _write_pair(out_clip_dir, 0, tir_path, rgb_path):
                counts[split] += 1
            else:
                skipped += 1
            total_seen += 1
            if total_seen % 1000 == 0:
                print(f"...{total_seen} frames processed")
    print(f"llvip: skipped {skipped} unreadable/missing pairs")
    print(f"llvip: train={counts['train']} val={counts['val']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["kaist", "llvip"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=5, help="kaist only")
    args = ap.parse_args()

    if args.dataset == "kaist":
        prep_kaist(args.src, args.out, args.stride)
    else:
        prep_llvip(args.src, args.out)


if __name__ == "__main__":
    main()
