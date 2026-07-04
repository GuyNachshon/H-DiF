import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


class RAFTFlow:
    """Wraps torchvision raft_large. flow_magnitude(a, b) -> mean |flow| float.
    Lazy weight load on first call."""

    def __init__(self, device="cpu"):
        self.device = device
        self._model = None
        self._transform = None

    def _ensure(self):
        if self._model is None:
            weights = Raft_Large_Weights.DEFAULT
            self._model = raft_large(weights=weights, progress=False).to(self.device).eval()
            self._transform = weights.transforms()

    @torch.no_grad()
    def flow_magnitude(self, frame_a, frame_b):
        self._ensure()
        a, b = self._transform(frame_a, frame_b)
        flow = self._model(a.to(self.device), b.to(self.device))[-1]
        return flow.norm(dim=1).mean().item()
