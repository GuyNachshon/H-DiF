# H-DiF: Hourglass Diffusion Flow Network

Thermal-to-RGB video colorization. H-DiF translates single-channel thermal
infrared (TIR) frames into photorealistic RGB using a Rectified Flow ODE on
top of a Hierarchical Hourglass Diffusion Transformer (HDiT), operating
natively in pixel space (no VAE). See `docs/research/PLAN.md` for the full
research plan and phase gates.

## Architecture

```
TIR frame + Canny edge map (2ch)
        │
   Patchify (4x4)
        │
 Outer transformer (local, high-res)  ──skip──┐
        │                                     │
 Inner bottleneck (global, low-res)           │
        │                                     │
   Attention residual injection (gamma * prev-frame attn)
        │
   Rectified velocity field v_theta(x_t, t)
        │
   Euler / midpoint ODE solver (1-4 steps)
        │
   RGB frame
```

- `src/models/hdit.py` — HDiT backbone (shifted-window + global attention levels).
- `src/flow/rectified.py` — rectified flow training loss and trajectory utilities.
- `src/sampling/ode.py` — Euler/midpoint ODE solvers for inference.
- `src/temporal/cache.py` — cross-frame attention residual cache (Phase 3).
- `src/data/paired.py` — paired TIR/RGB dataset loader.

## Quickstart

```bash
uv sync

# run tests
uv run pytest

# prepare data (KAIST HF mirror + LLVIP)
# download raw KAIST (kaist_train/, kaist_test/) and LLVIP (infrared/, visible/) locally, then:
uv run python scripts/prepare_data.py --dataset kaist --src /path/to/kaist_raw --out data/ --stride 5
uv run python scripts/prepare_data.py --dataset llvip --src /path/to/llvip_raw --out data/

# train
uv run python src/train.py --config config/rectified_flow.yaml --batch_size 16

# inference (video file or RTSP source -> mp4)
uv run python src/inference.py --checkpoint checkpoints/latest.safetensors --source input.mp4 --out output.mp4
```

Datasets are expected under `data/{train,val}/clips/<clip_id>/frame_%04d.{tir,rgb}.png`
after running `prepare_data.py`; see `src/data/paired.py` for the exact contract.

## Status

**Phase 1** (foundational HDiT backbone + pixel-space tokenization): in progress.
Success gate is edge-SSIM >= 0.90 between Canny edges of the input TIR frame and
the generated RGB, tracked as `val/edge_ssim` during training. Phases 2-4
(rectified flow matching, temporal attention residuals, multi-metric eval) are
scaffolded but not yet validated — see `docs/research/PLAN.md`.
