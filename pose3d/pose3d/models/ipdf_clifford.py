"""IPDF_ResNet: implicit-PDF scoring of rotor-rotated multivector features (Clifford Cl(3,0,0)).

Training scores the ground-truth rotor against random negative rotors (N-way cross-entropy);
inference searches a HEALPix SO(3) grid, optionally refined by gradient ascent.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from e3nn import o3
from image2sphere.so3_utils import so3_healpix_grid

from pose3d.models._utils import move_unregistered_tensors_to_device
from pose3d.models.ga_layers import TralaleroTralala


def _uniform_rotor_sample(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    u = torch.rand(n, 3, device=device, dtype=dtype)
    a = torch.sqrt(1 - u[:, 0]) * torch.sin(2 * math.pi * u[:, 1])
    b = torch.sqrt(1 - u[:, 0]) * torch.cos(2 * math.pi * u[:, 1])
    c = torch.sqrt(u[:, 0])      * torch.sin(2 * math.pi * u[:, 2])
    d = torch.sqrt(u[:, 0])      * torch.cos(2 * math.pi * u[:, 2])
    rotor = torch.zeros(n, 8, device=device, dtype=dtype)
    rotor[:, 0] = a   # scalar (e)
    rotor[:, 6] = b   # e23
    rotor[:, 5] = c   # e13
    rotor[:, 4] = d   # e12
    return rotor


def _bivector_exp(B: torch.Tensor) -> torch.Tensor:
    b12, b13, b23 = B[4], B[5], B[6]
    norm = torch.sqrt(b12**2 + b13**2 + b23**2 + 1e-10)
    s = torch.sin(norm) / norm
    result = torch.zeros_like(B)
    result[0] = torch.cos(norm)
    result[4] = s * b12
    result[5] = s * b13
    result[6] = s * b23
    return result


def _rotmat_to_rotor(R: torch.Tensor) -> torch.Tensor:
    m = R
    K = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    K[:, 0, 0] = m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2]
    K[:, 0, 1] = m[:, 0, 1] + m[:, 1, 0]
    K[:, 0, 2] = m[:, 0, 2] + m[:, 2, 0]
    K[:, 0, 3] = m[:, 2, 1] - m[:, 1, 2]
    K[:, 1, 1] = m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2]
    K[:, 1, 2] = m[:, 1, 2] + m[:, 2, 1]
    K[:, 1, 3] = m[:, 0, 2] - m[:, 2, 0]
    K[:, 2, 2] = m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1]
    K[:, 2, 3] = m[:, 1, 0] - m[:, 0, 1]
    K[:, 3, 3] = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    K = K + K.transpose(-1, -2) - torch.diag_embed(K.diagonal(dim1=-2, dim2=-1))
    K = K / 3.0
    _, eigvecs = torch.linalg.eigh(K)
    q_xyzw = eigvecs[:, :, -1]                          
    q_xyzw = q_xyzw * torch.sign(q_xyzw[:, 3:4] + 1e-10)  
    w  = q_xyzw[:, 3]
    qx = q_xyzw[:, 0]
    qy = q_xyzw[:, 1]
    qz = q_xyzw[:, 2]
    rotor = torch.zeros(R.shape[0], 8, device=R.device, dtype=R.dtype)
    rotor[:, 0] =  w   
    rotor[:, 4] = -qz   
    rotor[:, 5] = -qy  
    rotor[:, 6] = -qx   
    return rotor


def _rotor_to_rotmat(mv: torch.Tensor) -> torch.Tensor:
    w   =  mv[:, 0] 
    qz  = -mv[:, 4]  
    qy  = -mv[:, 5]   
    qx  = -mv[:, 6]  

    norm = torch.sqrt(w**2 + qx**2 + qy**2 + qz**2 + 1e-10)
    w, qx, qy, qz = w / norm, qx / norm, qy / norm, qz / norm

    R = torch.stack([
        1 - 2*(qy**2 + qz**2),  2*(qx*qy - w*qz),      2*(qx*qz + w*qy),
        2*(qx*qy + w*qz),        1 - 2*(qx**2 + qz**2),  2*(qy*qz - w*qx),
        2*(qx*qz - w*qy),        2*(qy*qz + w*qx),        1 - 2*(qx**2 + qy**2),
    ], dim=-1).reshape(-1, 3, 3)
    return R


class IPDF_ResNet(nn.Module):
    """Implicit-PDF style model using Clifford Algebra Cl(3,0,0).

    Training forward:
      1. Extract 256 multivectors from image via ResNet + conv_adapter.
      2. Rotate them by Q query rotors (GT at index 0, N_queries negatives)
         using the sandwich product  R·H·R̃.
      3. Score each rotated feature bag with a CGE-MLP (TralaleroTralala).
      4. Return logits [B, Q] — N-way CrossEntropy, target index = 0.

    Inference:
      - Grid search over a pre-computed SO(3) healpix grid.
      - Optional gradient refinement via bivector exponential map.
    """

    def __init__(
        self,
        algebra,
        hidden_dim: List = [32, 16],
        n_queries: int = 511,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        rec_level: int = 3,
    ):
        super().__init__()
        self.algebra   = algebra
        self.n_queries = n_queries

        self._mv_dim             = int(2 ** algebra.dim)
        self.conv_adapter_output = 16
        self._n_mv               = self.conv_adapter_output ** 2  # 256

        backbone_weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
        resnet = torchvision.models.resnet50(weights=backbone_weights)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.conv_adapter = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, self._mv_dim, kernel_size=1, bias=True),
            nn.AdaptiveAvgPool2d((self.conv_adapter_output, self.conv_adapter_output)),
        )

        self.scoring_head = TralaleroTralala(
            algebra=algebra,
            in_features=self._n_mv,
            hidden_dim=hidden_dim,
            out_features=1,
        )

        xyx         = so3_healpix_grid(rec_level=rec_level)
        grid_rotors = _rotmat_to_rotor(o3.angles_to_matrix(*xyx))
        self.register_buffer("so3_grid_rotors", grid_rotors, persistent=False)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_adapter(self.backbone(x)).flatten(2).transpose(1, 2)

    def _sandwich(self, R: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """R·H·R̃ for all (query, token) pairs.
        R : [B, Q, 8],  H : [B, N, 8]  ->  [B, Q, N, 8]
        """
        cayley = self.algebra.cayley
        R_rev  = self.algebra.beta(R)
        RH     = torch.einsum("bqi,ijk,bnk->bqnj",  R,  cayley, H)
        RHR    = torch.einsum("bqni,ijk,bqk->bqnj",  RH, cayley, R_rev)
        return RHR

    def _score(self, H: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """H : [B, N, 8],  queries : [B, Q, 8]  ->  logits [B, Q]"""
        B, Q, _ = queries.shape
        if getattr(self, "_extra_tensors_device", None) != H.device:
            move_unregistered_tensors_to_device(self, H.device)
            self._extra_tensors_device = H.device
        H_flat = self._sandwich(queries, H).reshape(B * Q, self._n_mv, 8)
        scored = self.scoring_head(H_flat)
        return scored[:, 0, 0].reshape(B, Q)

    def forward(self, x: torch.Tensor, rot_gt: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, Q].  GT rotor placed at query index 0."""
        B          = x.shape[0]
        H          = self._extract_features(x)
        gt_rotors  = _rotmat_to_rotor(rot_gt)
        neg_rotors = _uniform_rotor_sample(
            B * self.n_queries, device=x.device, dtype=x.dtype
        ).reshape(B, self.n_queries, 8)
        queries = torch.cat([gt_rotors.unsqueeze(1), neg_rotors], dim=1)
        return self._score(H, queries)

    def compute_loss(self, img: torch.Tensor, rot_gt: torch.Tensor, criterion=None) -> torch.Tensor:
        """N-way cross-entropy over [ground truth, negatives]; the target is always index 0."""
        logits = self.forward(img, rot_gt)
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, targets)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_batch(x)

    @torch.no_grad()
    def predict_batch(self, x: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
        """Grid search for a full batch.  ->  [B, 3, 3]"""
        B        = x.shape[0]
        H        = self._extract_features(x)
        G        = self.so3_grid_rotors.shape[0]
        best_idx = torch.zeros(B, dtype=torch.long, device=x.device)
        best_val = torch.full((B,), float("-inf"), device=x.device)

        for i in range(0, G, chunk_size):
            chunk          = self.so3_grid_rotors[i : i + chunk_size]
            logits         = self._score(H, chunk.unsqueeze(0).expand(B, -1, -1))
            local_max, idx = logits.max(dim=1)
            mask           = local_max > best_val
            best_val       = torch.where(mask, local_max, best_val)
            best_idx       = torch.where(mask, i + idx, best_idx)

        return _rotor_to_rotmat(self.so3_grid_rotors[best_idx])

    def predict_with_refinement(
        self, x: torch.Tensor, n_steps: int = 50, lr: float = 1e-2
    ) -> torch.Tensor:
        """Grid search + gradient refinement for a single image.  ->  [3, 3]"""
        with torch.no_grad():
            H              = self._extract_features(x)
            best_val, best_idx = float("-inf"), 0
            for i in range(0, self.so3_grid_rotors.shape[0], 256):
                v, idx = self._score(
                    H, self.so3_grid_rotors[i : i + 256].unsqueeze(0)
                )[0].max(0)
                if v.item() > best_val:
                    best_val, best_idx = v.item(), i + idx.item()

        R_best  = self.so3_grid_rotors[best_idx].clone()
        B_param = torch.zeros(8, device=x.device, dtype=x.dtype, requires_grad=True)
        opt     = torch.optim.Adam([B_param], lr=lr)

        for _ in range(n_steps):
            opt.zero_grad()
            R_current = self.algebra.geometric_product(
                _bivector_exp(B_param).unsqueeze(0), R_best.unsqueeze(0)
            ).squeeze(0)
            (-self._score(H, R_current.reshape(1, 1, 8))[0, 0]).backward()
            opt.step()

        with torch.no_grad():
            R_final = self.algebra.geometric_product(
                _bivector_exp(B_param).unsqueeze(0), R_best.unsqueeze(0)
            ).squeeze(0)
        return _rotor_to_rotmat(R_final.unsqueeze(0)).squeeze(0)
