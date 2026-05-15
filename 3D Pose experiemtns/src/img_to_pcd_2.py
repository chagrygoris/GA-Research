import torch
import torch.nn as nn
from transformers import AutoModelForDepthEstimation


def _infer_hidden_size_from_config(cfg):
    for k in ("hidden_size", "backbone_hidden_size", "d_model"):
        if hasattr(cfg, k):
            return getattr(cfg, k)
    raise ValueError("Could not infer hidden size from config")


class I2P(nn.Module):
    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Base-hf",
        layers: tuple = (-1, -3, -6, -9),
        n_points: int = 2048,
        freeze_backbone: bool = True,
        **kwargs
    ):
        super().__init__()
        self.layers, self.n_points = tuple(layers), n_points
        self.backbone = AutoModelForDepthEstimation.from_pretrained(model_name, output_hidden_states=True)
        self.backbone.config.output_hidden_states = True
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        d = 768 * len(self.layers)
        self.hs_mlp = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 256), nn.GELU())
        self.pt_mlp = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 256), nn.GELU())
        self.head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 9))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        out = self.backbone(pixel_values=x, output_hidden_states=True)

        hs = torch.cat([out.hidden_states[i].mean(dim=1) for i in self.layers], dim=-1)
        hs = self.hs_mlp(hs)

        depth = out.predicted_depth

        u, v = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij",
        )
        pts = torch.stack((v, u), dim=0).expand(b, -1, -1, -1)
        pts = torch.cat((pts, depth.unsqueeze(1)), dim=1).reshape(b, -1, 3)

        if pts.shape[1] > self.n_points:
            idx = torch.linspace(0, pts.shape[1] - 1, self.n_points, device=x.device).long()
            pts = pts[:, idx]

        pt = self.pt_mlp(pts).max(dim=1).values

        z = torch.cat((hs, pt), dim=-1)
        R = self.head(z).view(-1, 3, 3)
        return R