"""Zero-shot probe: do image-editing models preserve TIR structure without ever seeing thermal data?

Runs FLUX.1-Kontext-dev and/or Qwen-Image-Edit-2511 on 4 fixed KAIST val frames with a
"thermal -> daytime color photo" prompt, then scores edge_recall/colorfulness/lpips against
the visible-spectrum ground truth (same metrics as src/train.py's run_val).

GPU-only (24GB target): quantizes the transformer to NF4 via bitsandbytes and falls back to
CPU offload. Requires HUGGINGFACE_TOKEN env for gated FLUX.1-dev-family access — if the model
page hasn't been accepted, diffusers raises a 403/gated error; the script prints the model URL
to accept the license at and moves on.

Usage: uv run scripts/zeroshot_edit_probe.py --models kontext,qwenedit --out /workspace/probe_zeroshot
"""

import argparse
import gc
import os
import sys

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eval_metrics import colorfulness, edge_recall  # noqa: E402

_REPO = "koifisharriet/KAIST-Multispectral-Pedestrian-Benchmark"
_FRAMES = ["set06_V000_I00019", "set07_V000_I00019", "set08_V001_I00039", "set09_V000_I00019"]
_PROMPT = (
    "convert this thermal infrared photo into a realistic daytime color photograph of the same "
    "street scene, preserving the exact layout of roads, trees, buildings and vehicles"
)
_SIZE = 512


def _load_frames():
    """hf_hub_download the 4 fixed TIR+GT pairs, return list of (name, tir_pil_rgb, gt_pil_rgb)."""
    frames = []
    for name in _FRAMES:
        tir_path = hf_hub_download(_REPO, f"kaist_test/kaist_test_lwir/{name}_lwir.png", repo_type="dataset")
        gt_path = hf_hub_download(_REPO, f"kaist_test/kaist_test_visible/{name}_visible.png", repo_type="dataset")
        tir = Image.open(tir_path).convert("L").convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        frames.append((name, _resize_center_crop(tir), _resize_center_crop(gt)))
    return frames


def _resize_center_crop(img):
    w, h = img.size
    scale = _SIZE / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = img.size
    left, top = (w - _SIZE) // 2, (h - _SIZE) // 2
    return img.crop((left, top, left + _SIZE, top + _SIZE))


def _score(tir_pil, out_pil, gt_pil, lpips_net, device):
    """Mirror src/train.py's run_val metric conventions: TIR gray [0,1], pred/gt RGB [-1,1]."""
    tir_01 = to_tensor(tir_pil)[0].numpy()  # [H,W] in [0,1], grayscale channel
    out_11 = to_tensor(out_pil) * 2 - 1  # [3,H,W] in [-1,1]
    gt_11 = to_tensor(gt_pil) * 2 - 1
    out_np = out_11.permute(1, 2, 0).numpy()
    return {
        "edge_recall": edge_recall(tir_01, out_np),
        "colorfulness": colorfulness((out_np + 1) / 2),
        "lpips": lpips_net(out_11[None].to(device), gt_11[None].to(device)).item(),
    }


def _vram_peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30


def _try_nf4_config():
    """NF4 quant config for the transformer, or None if bitsandbytes isn't installed.

    bitsandbytes ships Linux-only wheels; this only runs on the RunPod GPU box.
    """
    try:
        import bitsandbytes  # noqa: F401
        from diffusers import BitsAndBytesConfig

        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    except ImportError:
        print("bitsandbytes not available -- loading transformer at full precision (may OOM).")
        return None


def run_kontext(frames, out_dir, device):
    from diffusers import FluxKontextPipeline, FluxTransformer2DModel

    model_id = "black-forest-labs/FLUX.1-Kontext-dev"
    quant_config = _try_nf4_config()
    try:
        kwargs = {"subfolder": "transformer", "torch_dtype": torch.bfloat16}
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
        transformer = FluxTransformer2DModel.from_pretrained(model_id, **kwargs)
        pipe = FluxKontextPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "403" in msg:
            print(f"kontext: gated model -- accept the license at https://huggingface.co/{model_id} and retry.")
        else:
            print(f"kontext: failed to load ({e}); skipping.")
        return None

    outputs = []
    for name, tir, gt in frames:
        image = pipe(image=tir, prompt=_PROMPT, guidance_scale=2.5, num_inference_steps=28).images[0]
        outputs.append((name, tir, image, gt))
    del pipe, transformer
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def run_qwenedit(frames, out_dir, device):
    from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel

    model_id = "Qwen/Qwen-Image-Edit-2511"
    quant_config = _try_nf4_config()
    try:
        kwargs = {"subfolder": "transformer", "torch_dtype": torch.bfloat16}
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
        transformer = QwenImageTransformer2DModel.from_pretrained(model_id, **kwargs)
        pipe = QwenImageEditPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
    except Exception as e:
        print(f"qwenedit: could not fit even quantized+offloaded ({e}); skipping.")
        return None

    outputs = []
    for name, tir, gt in frames:
        image = pipe(image=tir, prompt=_PROMPT, true_cfg_scale=2.5, num_inference_steps=28).images[0]
        outputs.append((name, tir, image, gt))
    del pipe, transformer
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


_MODELS = {"kontext": run_kontext, "qwenedit": run_qwenedit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="kontext,qwenedit")
    ap.add_argument("--out", default="/workspace/probe_zeroshot")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("warning: no CUDA device found -- this probe is meant to run on the GPU pod.")
    os.makedirs(args.out, exist_ok=True)

    import lpips as lpips_lib

    lpips_net = lpips_lib.LPIPS(net="vgg").to(device).eval().requires_grad_(False)

    frames = _load_frames()
    for model_name in args.models.split(","):
        model_name = model_name.strip()
        if model_name not in _MODELS:
            print(f"unknown model {model_name!r}; skipping.")
            continue
        print(f"=== {model_name} ===")
        torch.cuda.reset_peak_memory_stats()
        try:
            outputs = _MODELS[model_name](frames, args.out, device)
        except Exception as e:
            print(f"{model_name}: unexpected failure ({e}); skipping.")
            continue
        if outputs is None:
            continue

        rows = {"edge_recall": [], "colorfulness": [], "lpips": []}
        grid_tirs, grid_outs, grid_gts = [], [], []
        for name, tir, out_img, gt in outputs:
            m = _score(tir, out_img, gt, lpips_net, device)
            for k in rows:
                rows[k].append(m[k])
            print(f"  {name}: edge_recall={m['edge_recall']:.4f} colorfulness={m['colorfulness']:.4f} lpips={m['lpips']:.4f}")
            grid_tirs.append(to_tensor(tir))
            grid_outs.append(to_tensor(out_img))
            grid_gts.append(to_tensor(gt))

        n = len(outputs)
        print(f"  mean: edge_recall={sum(rows['edge_recall']) / n:.4f} "
              f"colorfulness={sum(rows['colorfulness']) / n:.4f} lpips={sum(rows['lpips']) / n:.4f}")
        print(f"  vram_peak_gb={_vram_peak_gb():.2f}")

        grid = make_grid(torch.stack(grid_tirs + grid_outs + grid_gts), nrow=n)
        grid_path = os.path.join(args.out, f"{model_name}.png")
        Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype("uint8")).save(grid_path)
        print(f"  grid saved to {grid_path}")


if __name__ == "__main__":
    main()
