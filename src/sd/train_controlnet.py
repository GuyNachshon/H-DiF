"""SD1.5 + ControlNet trainer: thermal+Canny hint -> RGB, single GPU, plain torch loop.

Mirrors src/train.py's house conventions (resume, checkpoint layout, HF upload, wandb,
non-finite-loss skip guard) but drives diffusers' SD1.5 UNet/ControlNet instead of the
scratch HDiT model.
"""

import argparse
import os
import time

import torch
import yaml
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from data.paired import PairedThermalRGB
from eval_metrics import colorfulness, edge_recall

_CKPT_DIR = "checkpoints_sd"


def swap_cond_channels(controlnet, n=2):
    """Replace the ControlNet hint conv_in with one accepting n input channels.

    Reads out_channels/kernel_size/padding off the existing conv so this tracks whatever
    conditioning_embedding_out_channels the pretrained net used, rather than hardcoding it.
    Kaiming-inits the new conv — simple and sufficient; no pretrained-weight transplant.
    """
    old_conv = controlnet.controlnet_cond_embedding.conv_in
    new_conv = torch.nn.Conv2d(
        n, old_conv.out_channels, kernel_size=old_conv.kernel_size, padding=old_conv.padding
    )
    torch.nn.init.kaiming_normal_(new_conv.weight, nonlinearity="relu")
    torch.nn.init.zeros_(new_conv.bias)
    controlnet.controlnet_cond_embedding.conv_in = new_conv
    return controlnet


def _atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(controlnet, opt, step, wandb_run_id):
    step_dir = os.path.join(_CKPT_DIR, f"step_{step}")
    controlnet.save_pretrained(step_dir)
    controlnet.save_pretrained(os.path.join(_CKPT_DIR, "latest"))
    _atomic_torch_save(
        {"opt": opt.state_dict(), "step": step, "wandb_run_id": wandb_run_id},
        os.path.join(_CKPT_DIR, "latest.pt"),
    )


def load_checkpoint(controlnet, opt, device):
    latest = os.path.join(_CKPT_DIR, "latest.pt")
    latest_dir = os.path.join(_CKPT_DIR, "latest")
    if not os.path.exists(latest) or not os.path.isdir(latest_dir):
        return 0, None
    try:
        state = torch.load(latest, map_location=device, weights_only=True)
        loaded = ControlNetModel.from_pretrained(latest_dir)
        controlnet.load_state_dict(loaded.state_dict())
        opt.load_state_dict(state["opt"])
        return state["step"], state.get("wandb_run_id")
    except Exception as e:
        print(f"warning: corrupt checkpoint, starting fresh — inspect {_CKPT_DIR}/ manually ({e})")
        return 0, None


def upload_checkpoint(repo_id, step):
    if not os.environ.get("HUGGINGFACE_TOKEN"):
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
        api.create_repo(repo_id, exist_ok=True, private=True)
        api.upload_folder(
            folder_path=os.path.join(_CKPT_DIR, "latest"),
            path_in_repo=f"step_{step}",
            repo_id=repo_id,
        )
        api.upload_folder(
            folder_path=os.path.join(_CKPT_DIR, "latest"),
            path_in_repo="latest",
            repo_id=repo_id,
        )
    except Exception as e:
        print(f"warning: HF upload failed: {e}")


def _wandb_log(payload, step):
    try:
        import wandb

        wandb.log(payload, step=step)
    except Exception as e:
        print(f"warning: wandb.log failed at step {step}: {e}")


@torch.no_grad()
def run_val(pipe, val_dl, sampling_cfg, val_batches, device, use_wandb, step, weight_dtype):
    """Sample RGB from val TIR via the live pipeline, score against ground truth.

    Same metrics as src/train.py's run_val: edge_recall, colorfulness (+gt), lpips.
    """
    import lpips as lpips_lib

    lpips_net = lpips_lib.LPIPS(net="vgg").to(device).eval().requires_grad_(False)
    pipe.controlnet.eval()
    ssims, edge_recalls, colorfulnesses, colorfulnesses_gt, lpips_scores = [], [], [], [], []
    samples = None
    for i, batch in enumerate(val_dl):
        if i >= val_batches:
            break
        cond = batch["tir"].to(device, dtype=weight_dtype)
        x1 = batch["rgb"].to(device)
        bsz = cond.shape[0]

        def _sample(seed):
            gen = torch.Generator(device=device).manual_seed(seed)
            out = pipe(
                prompt=[""] * bsz,
                image=cond,
                num_inference_steps=sampling_cfg["num_inference_steps"],
                guidance_scale=sampling_cfg["guidance_scale"],
                generator=gen,
                output_type="pt",
            ).images
            return out.to(device).float() * 2 - 1  # pipe returns [0,1] -> match x1's [-1,1]

        x_pred = _sample(0)
        gt_np = x1.permute(0, 2, 3, 1).cpu().numpy()
        pred_np = x_pred.permute(0, 2, 3, 1).cpu().numpy()
        tir_np = cond[:, 0].float().cpu().numpy()
        lpips_scores.append(lpips_net(x_pred.clamp(-1, 1), x1).mean().item())
        for t, p, g in zip(tir_np, pred_np, gt_np):
            ssims.append(structural_similarity(g, p, channel_axis=2, data_range=2.0))
            edge_recalls.append(edge_recall(t, p))
            colorfulnesses.append(colorfulness((p + 1) / 2))
            colorfulnesses_gt.append(colorfulness((g + 1) / 2))
        if i == 0 and use_wandb:
            x_pred2 = _sample(1)
            samples = (cond[:8, 0:1], x_pred[:8], x_pred2[:8], x1[:8])

    pipe.controlnet.train()
    ssim = sum(ssims) / len(ssims)
    er = sum(edge_recalls) / len(edge_recalls)
    cf = sum(colorfulnesses) / len(colorfulnesses)
    cf_gt = sum(colorfulnesses_gt) / len(colorfulnesses_gt)
    lp = sum(lpips_scores) / len(lpips_scores)
    print(f"step {step} val/ssim {ssim:.4f} val/edge_recall {er:.4f} val/colorfulness {cf:.4f} "
          f"val/colorfulness_gt {cf_gt:.4f} val/lpips {lp:.4f}")
    if use_wandb:
        import wandb
        from torchvision.utils import make_grid

        tir3, pred0, pred1, gt = samples
        tir3 = tir3.float().repeat(1, 3, 1, 1)
        grid_rows = torch.cat(
            [tir3, (pred0.clamp(-1, 1) + 1) / 2, (pred1.clamp(-1, 1) + 1) / 2, (gt + 1) / 2], dim=0
        )
        grid = make_grid(grid_rows, nrow=tir3.shape[0])
        _wandb_log(
            {
                "val/ssim": ssim,
                "val/edge_recall": er,
                "val/colorfulness": cf,
                "val/colorfulness_gt": cf_gt,
                "val/lpips": lp,
                "val/samples": wandb.Image(grid, caption="rows: TIR / pred_seed0 / pred_seed1 / gt"),
            },
            step,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg["train"]
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
    if args.steps is not None:
        train_cfg["steps"] = args.steps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = cfg["model"]["base"]

    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=weight_dtype).to(device)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=weight_dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=weight_dtype).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    unet.requires_grad_(False).eval()

    controlnet = ControlNetModel.from_pretrained(cfg["model"]["controlnet_init"], torch_dtype=weight_dtype).to(device)
    swap_cond_channels(controlnet, n=2)
    controlnet.to(device, dtype=weight_dtype)
    controlnet.train()

    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()

    with torch.no_grad():
        empty_ids = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
        empty_emb = text_encoder(empty_ids)[0]  # [1, L, D], cached and expanded per-batch below

    num_workers = 8
    ds = PairedThermalRGB(cfg["data"]["root"], "train", size=cfg["data"]["size"], clip_len=cfg["data"].get("clip_len", 1))
    dl = DataLoader(
        ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_dl = None
    try:
        val_ds = PairedThermalRGB(cfg["data"]["root"], "val", size=cfg["data"]["size"], clip_len=cfg["data"].get("clip_len", 1))
        val_dl = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=2)
    except FileNotFoundError:
        print("warning: no val split found, skipping validation")

    opt = torch.optim.AdamW(
        controlnet.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )

    os.makedirs(_CKPT_DIR, exist_ok=True)
    step, wandb_run_id = load_checkpoint(controlnet, opt, device)
    if step > 0:
        print(f"resumed from step {step}")

    use_wandb = train_cfg.get("wandb", False)
    if use_wandb:
        import wandb

        run_name = f"{os.path.splitext(os.path.basename(args.config))[0]}-{int(time.time())}"
        wandb.init(project="h-dif", config=cfg, name=run_name, id=wandb_run_id, resume="allow")
        wandb_run_id = wandb.run.id

    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)

    hf_repo = train_cfg.get("hf_repo")
    grad_accum = train_cfg["grad_accum"]
    skipped_steps = 0
    last_log_time = time.time()
    last_val_step = None
    autocast_enabled = torch.cuda.is_available()
    opt.zero_grad()
    accum_count = 0
    while step < train_cfg["steps"]:
        for batch in dl:
            cond = batch["tir"].to(device, dtype=weight_dtype)
            rgb = batch["rgb"].to(device, dtype=weight_dtype)
            bsz = cond.shape[0]

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                latents = vae.encode(rgb).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = empty_emb.expand(bsz, -1, -1)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                down_res, mid_res = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=cond,
                    return_dict=False,
                )
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                    return_dict=False,
                )[0]
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float()) / grad_accum

            if not torch.isfinite(loss):
                skipped_steps += 1
                step += 1
                opt.zero_grad()
                accum_count = 0
                continue

            loss.backward()
            accum_count += 1
            if accum_count < grad_accum:
                step += 1
                if step >= train_cfg["steps"]:
                    break
                continue
            accum_count = 0

            grad_norm = torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            if step % train_cfg["log_every"] == 0:
                steps_per_sec = train_cfg["log_every"] / max(time.time() - last_log_time, 1e-8)
                last_log_time = time.time()
                print(f"step {step} loss {loss.item() * grad_accum:.4f} grad_norm {grad_norm:.4f} skipped {skipped_steps}")
                if use_wandb:
                    _wandb_log(
                        {
                            "loss": loss.item() * grad_accum,
                            "lr": opt.param_groups[0]["lr"],
                            "grad_norm": grad_norm.item(),
                            "steps_per_sec": steps_per_sec,
                            "skipped_steps": skipped_steps,
                        },
                        step,
                    )
            if step > 0 and step % train_cfg["save_every"] == 0:
                save_checkpoint(controlnet, opt, step, wandb_run_id)
                if hf_repo:
                    upload_checkpoint(hf_repo, step)
            if val_dl is not None and step > 0 and step % train_cfg["val_every"] == 0:
                run_val(pipe, val_dl, cfg["sampling"], train_cfg["val_batches"], device, use_wandb, step, weight_dtype)
                last_val_step = step
            step += 1
            if step >= train_cfg["steps"]:
                break

    if step > 0:
        if val_dl is not None and step != last_val_step:
            run_val(pipe, val_dl, cfg["sampling"], train_cfg["val_batches"], device, use_wandb, step, weight_dtype)
        save_checkpoint(controlnet, opt, step, wandb_run_id)
        if hf_repo:
            upload_checkpoint(hf_repo, step)


if __name__ == "__main__":
    main()
