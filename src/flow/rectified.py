import torch
import torch.nn.functional as F
from torch import nn

_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(2, 3).contiguous()


def _grad_mag(img):
    """[B,3,H,W] in [-1,1] -> luma Sobel gradient magnitude [B,1,H,W]."""
    luma = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    gx = F.conv2d(luma, _SOBEL_X.to(img.device, img.dtype), padding=1)
    gy = F.conv2d(luma, _SOBEL_Y.to(img.device, img.dtype), padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def make_x0(cond, mode="noise", alpha=0.7):
    """Starting point x0 for the flow, matching cond's RGB shape (B,3,H,W).

    - "noise": pure Gaussian noise (best color/diversity, loose structure).
    - "tir": TIR channel broadcast to 3ch (best structure, color collapse).
    - "blend": alpha*noise + (1-alpha)*tir, interpolating the two.
    """
    tir3 = cond[:, 0:1].repeat(1, 3, 1, 1)
    if mode == "tir":
        return tir3
    noise = torch.randn_like(tir3)
    if mode == "noise":
        return noise
    if mode == "blend":
        return alpha * noise + (1 - alpha) * tir3
    raise ValueError(f"unknown x0_mode: {mode}")


def flow_batch(x0, x1, t):
    """x_t = (1-t)*x0 + t*x1 ; v = x1 - x0. t: [B] broadcast to [B,1,1,1]."""
    t = t.view(-1, 1, 1, 1)
    x_t = (1 - t) * x0 + t * x1
    v = x1 - x0
    return x_t, v


# Sobel mag of a unit step edge is ~4 (kernel abs-sum); this constant rescales _grad_mag's
# raw (unnormalized) output so a strong edge lands near 1.0 for the SGA asymmetric term.
_SGA_EDGE_SCALE = 0.25


class RectifiedFlow(nn.Module):
    def __init__(
        self,
        model,
        t_sampling="uniform",
        cond_dropout=0.0,
        lpips_weight=0.0,
        edge_mode="none",
        edge_weight=0.0,
    ):
        super().__init__()
        self.model = model
        self.t_sampling = t_sampling
        self.cond_dropout = cond_dropout
        self.lpips_weight = lpips_weight
        if edge_mode not in ("none", "sga"):
            raise ValueError(f"unknown edge_mode: {edge_mode}")
        self.edge_mode = edge_mode
        self.edge_weight = edge_weight
        self._lpips = None  # lazy-init on first use, see _lpips_net()

    def _lpips_net(self, device):
        if self._lpips is None:
            import lpips

            self._lpips = lpips.LPIPS(net="vgg").to(device).eval().requires_grad_(False)
        return self._lpips

    def forward(self, x_t, t, cond, cache=None):
        x = torch.cat([x_t, cond], dim=1)
        # t is the flow time in [0,1]; the model reads it as sigma (log(sigma)/4 internally),
        # so clamp away from 0 to keep the log finite at the trajectory start.
        sigma = t.clamp_min(1e-4)
        return self.model(x, sigma, cache=cache)

    def loss(self, x0, x1, cond, t=None):
        if t is None:
            if self.t_sampling == "logit_normal":
                t = torch.sigmoid(torch.randn(x0.shape[0], device=x0.device))
            else:
                t = torch.rand(x0.shape[0], device=x0.device)
        if self.cond_dropout > 0:
            keep = torch.rand(x0.shape[0], 1, 1, 1, device=x0.device) >= self.cond_dropout
            cond = torch.where(keep, cond, torch.zeros_like(cond))
        x_t, v = flow_batch(x0, x1, t)
        v_pred = self(x_t, t, cond)
        total_loss = torch.mean((v_pred - v) ** 2)

        t_col = t.view(-1, 1, 1, 1)
        trustworthy = t_col.squeeze() > 0.5  # x1_hat only reliable near the clean (t=1) end
        if trustworthy.any() and (self.lpips_weight > 0 or self.edge_weight > 0):
            x1_hat = x_t + (1 - t_col) * v_pred

        if self.lpips_weight > 0 and trustworthy.any():
            # VGG activations at full batch/res OOM a 24GB card (bs64 x 256^2 killed
            # the first probe run): cap samples and evaluate at 128^2 — standard
            # practice, signal is preserved.
            x1_hat_m = x1_hat[trustworthy][:16].clamp(-1, 1)
            x1_m = x1[trustworthy][:16]
            if x1_m.shape[-1] > 128:
                x1_hat_m = F.interpolate(x1_hat_m, size=128, mode="bilinear", align_corners=False)
                x1_m = F.interpolate(x1_m, size=128, mode="bilinear", align_corners=False)
            net = self._lpips_net(x0.device)
            lpips_loss = net(x1_hat_m, x1_m).mean()
            total_loss = total_loss + self.lpips_weight * lpips_loss

        if self.edge_mode == "sga" and self.edge_weight > 0 and trustworthy.any():
            edge_pred_normfree = _grad_mag(x1_hat[trustworthy]) * _SGA_EDGE_SCALE
            tir_edge = cond[trustworthy, 1:2]
            tir_edge_mask = tir_edge > 0.5
            if tir_edge_mask.any():
                sga_term = F.relu(tir_edge - edge_pred_normfree).pow(2)
                edge_loss = sga_term[tir_edge_mask].mean()
                total_loss = total_loss + self.edge_weight * edge_loss

        return total_loss


@torch.no_grad()
def trajectory_straightness(rf, x0, x1, cond, n_probes=8):
    """Variance of v_pred over t at fixed endpoints -> float; lower = straighter."""
    preds = []
    for i in range(n_probes):
        t = torch.full((x0.shape[0],), (i + 0.5) / n_probes, device=x0.device)
        x_t, _ = flow_batch(x0, x1, t)
        preds.append(rf(x_t, t, cond))
    preds = torch.stack(preds, dim=0)
    return preds.var(dim=0).mean().item()
