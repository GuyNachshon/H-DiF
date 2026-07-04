import argparse
import os

import torch
import yaml
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from data.paired import PairedThermalRGB
from flow.rectified import RectifiedFlow
from models.build import build_hdit


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

    ds = PairedThermalRGB(cfg["data"]["root"], "train", size=cfg["data"]["size"], clip_len=cfg["data"].get("clip_len", 1))
    dl = DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=4, drop_last=True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg["betas"]),
    )

    use_wandb = train_cfg.get("wandb", False)
    if use_wandb:
        import wandb

        wandb.init(project="h-dif", config=cfg)

    os.makedirs("checkpoints", exist_ok=True)
    step = 0
    # ponytail: EMA/resume/schedulers land with real training — scaffold loop only.
    while step < train_cfg["steps"]:
        for batch in dl:
            cond = batch["tir"].to(device)
            x1 = batch["rgb"].to(device)
            x0 = cond[:, 0:1].repeat(1, 3, 1, 1)
            loss = rf.loss(x0, x1, cond)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % train_cfg["log_every"] == 0:
                print(f"step {step} loss {loss.item():.4f}")
                if use_wandb:
                    wandb.log({"loss": loss.item()}, step=step)
            if step > 0 and step % train_cfg["save_every"] == 0:
                save_file(model.state_dict(), f"checkpoints/step_{step}.safetensors")
            step += 1
            if step >= train_cfg["steps"]:
                break


if __name__ == "__main__":
    main()
