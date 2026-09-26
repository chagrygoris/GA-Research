"""Matrix Fisher prior for CliffordFlow's source rotation r0 (experimental, feature `fisher_prior`).

Distribution maths ported from Liu et al. 2023 (github.com/PKU-EPIC/RotationNormFlow,
utils/fisher.py), dropping the pytorch3d/nflows/scipy dependencies. The ResNet-101 head
mirrors their Pascal3D+ checkpoint layout so `state_dict_119.pkl` loads unchanged.
"""

import numpy as np
import torch
import torch.nn as nn


def quat_to_rotmat(quat):
    quat = quat / quat.norm(p=2, dim=-1, keepdim=True)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    w2, x2, y2, z2 = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R = torch.stack([
        w2 + x2 - y2 - z2, 2 * (xy - wz), 2 * (wy + xz),
        2 * (wz + xy), w2 - x2 + y2 - z2, 2 * (yz - wx),
        2 * (xz - wy), 2 * (wx + yz), w2 - x2 - y2 + z2,
    ], dim=-1).view(*quat.shape[:-1], 3, 3)
    return R


def proper_svd(A):
    U, S, Vh = torch.linalg.svd(A)
    detU, detV = torch.linalg.det(U), torch.linalg.det(Vh)
    U, S, Vh = U.clone(), S.clone(), Vh.clone()
    U[..., :, 2] *= detU.unsqueeze(-1)
    S[..., 2] *= detU * detV
    Vh[..., 2, :] *= detV.unsqueeze(-1)
    return U, S, Vh


def norm_approx(S):
    return 1.0 / torch.sqrt(8 * torch.pi * (S[..., 0] + S[..., 1]) * (S[..., 1] + S[..., 2]) * (S[..., 0] + S[..., 2]))


def log_prob(A, R):
    _, S, _ = proper_svd(A)
    trace = (R * A).sum(-1).sum(-1)
    return trace - S.sum(-1) - norm_approx(S).log()


def _sample_bingham(A4, Omega, std, M_star, n, oversample=8):
    device = A4.device
    while True:
        eps = torch.randn(n * oversample, 4, device=device)
        y = std * eps
        s = y / y.norm(dim=-1, keepdim=True)
        p_bing = torch.exp(-(s * A4 * s).sum(-1))
        p_acg = ((s * Omega * s).sum(-1)) ** (-2)
        accept = torch.rand(n * oversample, device=device) < p_bing / (M_star * p_acg)
        if accept.sum() >= n:
            return s[accept][:n]


def sample_one(A, b=1.5, oversample=8):
    # A: (3, 3) -> single (3, 3) rotation. Rejection sampling isn't
    # differentiable, so this always runs on a detached A.
    U, S, Vh = proper_svd(A.detach())

    A4 = torch.zeros(4, device=A.device, dtype=A.dtype)
    A4[1] = 2 * (S[1] + S[2])
    A4[2] = 2 * (S[0] + S[2])
    A4[3] = 2 * (S[0] + S[1])

    Omega = 1 + 2 * A4 / b
    std = Omega ** -0.5
    M_star = np.exp(-(4 - b) / 2) * (4 / b) ** 2

    quat = _sample_bingham(A4, Omega, std, M_star, n=1, oversample=oversample)
    R = quat_to_rotmat(quat)[0]
    return U @ R @ Vh


@torch.no_grad()
def sample_batch(A):
    # A: (N, 3, 3) -> (N, 3, 3), one sample per row. Used only as r0, an
    # initial guess drawn from a frozen, pretrained Fisher head.
    return torch.stack([sample_one(a) for a in A], dim=0)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, layers=(3, 4, 23, 3)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, 1000)  # unused, only for pretrained load
        self.output_size = 512 * Bottleneck.expansion

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def resnet101():
    return ResNet(layers=(3, 4, 23, 3))


class ResnetHead(nn.Module):
    def __init__(self, base, n_classes, embedding_dim, num_hidden_nodes, n_out):
        super().__init__()
        self.base = base
        self.class_embedding = nn.Embedding(n_classes, embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(base.output_size + embedding_dim, num_hidden_nodes),
            nn.BatchNorm1d(num_hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.BatchNorm1d(num_hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_nodes, n_out),
        )

    def forward(self, img, class_idx):
        latent = self.base(img)
        conc = torch.cat([latent, self.class_embedding(class_idx)], dim=1)
        return conc, self.head(conc)
