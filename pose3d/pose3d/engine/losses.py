"""Loss functions for the direct-regression / rotor-output models.

CliffordFlow, IPDF and the I2S/i2s_real heads compute their own loss in `compute_loss`
and ignore the criterion built from these.
"""

import torch
import torch.nn as nn

from pose3d.geometry.quaternion import matrix_to_unit_quaternion


def rotation_matrix_loss(pred, target, alpha=0.1, beta=0.01):
    mse = torch.mean((pred - target) ** 2)

    eye = torch.eye(3, device=pred.device).unsqueeze(0)
    ortho = torch.mean((pred.transpose(1, 2) @ pred - eye) ** 2)

    return mse + alpha * ortho


def project_to_rotation_matrix(x: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(x)
    r = u @ vh

    det = torch.det(r)

    correction = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1)
    correction[:, -1, -1] = torch.where(
        det < 0,
        torch.tensor(-1.0, device=x.device, dtype=x.dtype),
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
    )

    return u @ correction @ vh


def geodesic_rotation_matrix_loss(pred, target, alpha=0.01, beta=0.001):
    pred_rot = project_to_rotation_matrix(pred)

    relative = pred_rot.transpose(1, 2) @ target

    trace = relative[:, 0, 0] + relative[:, 1, 1] + relative[:, 2, 2]
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = cos_theta.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    angle_loss = torch.acos(cos_theta).mean()

    eye = torch.eye(3, device=pred.device, dtype=pred.dtype).unsqueeze(0)
    ortho = torch.mean((pred.transpose(1, 2) @ pred - eye) ** 2)

    det_loss = torch.mean((torch.det(pred) - 1.0) ** 2)

    return angle_loss + alpha * ortho + beta * det_loss


def rotor_loss(pred: torch.Tensor, target_rot: torch.Tensor) -> torch.Tensor:
    target = matrix_to_unit_quaternion(target_rot)
    pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    dot = torch.sum(pred * target, dim=-1).abs()
    loss = 1.0 - dot.pow(2)
    return loss.mean()


def multivector_rotor_loss(
    pred_mv,
    target_rot,
    lambda_non_even=0.0,
    lambda_norm=0.0,
):
    target = matrix_to_unit_quaternion(target_rot)

    pred = pred_mv[:, [0, 4, 5, 6]]
    pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    dot = torch.sum(pred * target, dim=-1).abs()
    return (1.0 - dot.pow(2)).mean()


def build_criterion(cfg):
    loss = cfg.train.loss
    if loss == "mse":
        return nn.MSELoss()
    if loss == "mse_ortho":
        return rotation_matrix_loss
    if loss == "geodesic":
        return geodesic_rotation_matrix_loss
    if loss == "prob":
        return nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    if loss == "rotor":
        return rotor_loss
    if loss == "mv_rotor":
        return multivector_rotor_loss
    raise ValueError(f"Unknown loss: {loss}")
