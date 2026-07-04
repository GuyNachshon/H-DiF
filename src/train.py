import argparse
import os
import time

import cv2
import torch
import yaml
from safetensors.torch import load_file, save_file
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from data.paired import PairedThermalRGB
from flow.rectified import RectifiedFlow
from models.build import build_hdit
from sampling.ode import euler, midpoint

_CANNY_THRESH = (100, 200)  # matches data/paired.py and inference.py

_SOLVERS = {"euler": euler, "midpoint": midpoint}
_CKPT_DIR = "checkpoints"


def save_checkpoint(model, opt, step, wandb_run_id):
    save_file(model.state_dict(), os.path.join(_CKPT_DIR, f"step_{step}.safetensors"))
    torch.save(
        {"opt": opt.state_dict(), "step": step, "wandb_run_id": wandb_run_id},
        os.path.join(_CKPT_DIR, "latest.pt"),
    )
    # ponytail: "latest.pt" always points at the most recent save; weights file is named by step for HF history.
    save_file(model.state_dict(), os.path.join(_CKPT_DIR, "latest.safetensors"))


def load_checkpoint(model, opt, device):
    latest = os.path.join(_CKPT_DIR, "latest.pt")
    if not os.path.exists(latest):
        return 0, None
    state = torch.load(latest, map_location=device, weights_only=True)
    model.load_state_dict(load_file(os.path.join(_CKPT_DIR, "latest.safetensors")))
    opt.load_state_dict(state["opt"])
    return state["step"], state.get("wandb_run_id")


def upload_checkpoint(repo_id, step):
    if not os.environ.get("HUGGINGFACE_TOKEN"):
        return
    try:
        from huggingface_hub import HfApi

        # huggingface_hub only auto-reads HF_TOKEN, not HUGGINGFACE_TOKEN
        api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
        api.create_repo(repo_id, exist_ok=True, private=True)
        for fname in (f"step_{step}.safetensors", "latest.pt", "latest.safetensors"):
            api.upload_file(
                path_or_fileobj=os.path.join(_CKPT_DIR, fname),
                path_in_repo=fname,
                repo_id=repo_id,
            )
    except Exception as e:
        print(f"warning: HF upload failed: {e}")


def _edge_ssim(tir_np, pred_np):
    """Canny-edge SSIM between input TIR channel and predicted RGB (Phase-1 gate, PLAN.md 1.4)."""
    tir_u8 = (tir_np * 255.0).clip(0, 255).astype("uint8")
    pred_gray = cv2.cvtColor(((pred_np + 1.0) * 127.5).clip(0, 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
    edges_tir = cv2.Canny(tir_u8, *_CANNY_THRESH)
    edges_pred = cv2.Canny(pred_gray, *_CANNY_THRESH)
    return structural_similarity(edges_tir, edges_pred, data_range=255)


@torch.no_grad()
def run_val(rf, val_dl, sampling_cfg, val_batches, device, use_wandb, step):
    """Sample RGB from val TIR, score against ground truth. Prints + logs val/ssim, val/mse, val/edge_ssim."""
    solver = _SOLVERS[sampling_cfg["solver"]]
    rf.eval()
    ssims, mses, edge_ssims = [], [], []
    samples = None
    for i, batch in enumerate(val_dl):
        if i >= val_batches:
            break
        cond = batch["tir"].to(device)
        x1 = batch["rgb"].to(device)
        x0 = cond[:, 0:1].repeat(1, 3, 1, 1)
        x_pred = solver(rf, x0, cond, steps=sampling_cfg["steps"])
        mses.append(torch.mean((x_pred - x1) ** 2).item())
        tir_np = cond[:, 0].cpu().numpy()
        pred_np = x_pred.permute(0, 2, 3, 1).cpu().numpy()
        gt_np = x1.permute(0, 2, 3, 1).cpu().numpy()
        for t, p, g in zip(tir_np, pred_np, gt_np):
            ssims.append(structural_similarity(g, p, channel_axis=2, data_range=2.0))
            edge_ssims.append(_edge_ssim(t, p))
        if i == 0 and use_wandb:
            samples = (cond[:8, 0:1], x_pred[:8], x1[:8])
    rf.train()
    ssim, mse = sum(ssims) / len(ssims), sum(mses) / len(mses)
    edge_ssim = sum(edge_ssims) / len(edge_ssims)
    print(f"step {step} val/ssim {ssim:.4f} val/mse {mse:.4f} val/edge_ssim {edge_ssim:.4f}")
    if use_wandb:
        import wandb
        from torchvision.utils import make_grid

        tir3, pred, gt = samples
        tir3 = tir3.repeat(1, 3, 1, 1)  # grayscale -> 3ch for the grid
        grid_rows = torch.cat([tir3, (pred.clamp(-1, 1) + 1) / 2, (gt + 1) / 2], dim=0)
        grid = make_grid(grid_rows, nrow=tir3.shape[0])
        wandb.log(
            {
                "val/ssim": ssim,
                "val/mse": mse,
                "val/edge_ssim": edge_ssim,
                "val/samples": wandb.Image(grid, caption="rows: TIR / pred / gt"),
            },
            step=step,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg["train"]
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = dict(cfg["model"], size=cfg["data"]["size"])
    model = build_hdit(model_cfg).to(device)
    rf = RectifiedFlow(model).to(device)

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
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg["betas"]),
    )

    os.makedirs(_CKPT_DIR, exist_ok=True)
    step, wandb_run_id = load_checkpoint(model, opt, device)
    if step > 0:
        print(f"resumed from step {step}")

    use_wandb = train_cfg.get("wandb", False)
    if use_wandb:
        import wandb

        run_name = f"{os.path.splitext(os.path.basename(args.config))[0]}-{int(time.time())}"
        wandb.init(project="h-dif", config=cfg, name=run_name, id=wandb_run_id, resume="allow")
        wandb_run_id = wandb.run.id

    hf_repo = train_cfg.get("hf_repo")
    autocast_enabled = torch.cuda.is_available()
    skipped_steps = 0
    last_log_time = time.time()
    # ponytail: EMA/schedulers still land with real training — scaffold loop only.
    while step < train_cfg["steps"]:
        for batch in dl:
            cond = batch["tir"].to(device)
            x1 = batch["rgb"].to(device)
            x0 = cond[:, 0:1].repeat(1, 3, 1, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss = rf.loss(x0, x1, cond)

            if not torch.isfinite(loss):
                skipped_steps += 1
                step += 1
                continue

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % train_cfg["log_every"] == 0:
                steps_per_sec = train_cfg["log_every"] / max(time.time() - last_log_time, 1e-8)
                last_log_time = time.time()
                print(f"step {step} loss {loss.item():.4f} grad_norm {grad_norm:.4f} skipped {skipped_steps}")
                if use_wandb:
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "lr": opt.param_groups[0]["lr"],
                            "grad_norm": grad_norm.item(),
                            "steps_per_sec": steps_per_sec,
                            "skipped_steps": skipped_steps,
                        },
                        step=step,
                    )
            if step > 0 and step % train_cfg["save_every"] == 0:
                save_checkpoint(model, opt, step, wandb_run_id)
                if hf_repo:
                    upload_checkpoint(hf_repo, step)
            if val_dl is not None and step > 0 and step % train_cfg["val_every"] == 0:
                run_val(rf, val_dl, cfg["sampling"], train_cfg["val_batches"], device, use_wandb, step)
            step += 1
            if step >= train_cfg["steps"]:
                break


if __name__ == "__main__":
    main()
