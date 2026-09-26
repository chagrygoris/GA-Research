"""Unit-quaternion / rotor-multivector decoding shared by the loss and evaluation code."""

import torch


def unit_quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    if q.ndim != 2 or q.shape[-1] != 4:
        raise ValueError(f"Expected unit quaternion shape [B, 4], got {tuple(q.shape)}")
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        [
            torch.stack((ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)), dim=-1),
            torch.stack((2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)), dim=-1),
            torch.stack((2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz), dim=-1),
        ],
        dim=-2,
    )


def project_multivector_to_rotor(mv: torch.Tensor) -> torch.Tensor:
    if mv.shape[-1] != 8:
        raise ValueError(
            "project_multivector_to_rotor currently supports only Cl(3,0), "
            f"expected mv_dim=8, got {mv.shape[-1]}"
        )
    rotor = mv[:, [0, 4, 5, 6]]
    rotor = rotor / rotor.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return rotor


def matrix_to_unit_quaternion(rot: torch.Tensor) -> torch.Tensor:
    if rot.ndim != 3 or rot.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices with shape [B, 3, 3], got {tuple(rot.shape)}")

    m00 = rot[:, 0, 0]
    m01 = rot[:, 0, 1]
    m02 = rot[:, 0, 2]
    m10 = rot[:, 1, 0]
    m11 = rot[:, 1, 1]
    m12 = rot[:, 1, 2]
    m20 = rot[:, 2, 0]
    m21 = rot[:, 2, 1]
    m22 = rot[:, 2, 2]
    trace = m00 + m11 + m22

    q = torch.zeros((rot.shape[0], 4), device=rot.device, dtype=rot.dtype)
    eps = 1e-8

    mask_trace = trace > 0.0
    if mask_trace.any():
        t = torch.sqrt((trace[mask_trace] + 1.0).clamp_min(eps)) * 2.0
        q[mask_trace, 0] = 0.25 * t
        q[mask_trace, 1] = (m21[mask_trace] - m12[mask_trace]) / t
        q[mask_trace, 2] = (m02[mask_trace] - m20[mask_trace]) / t
        q[mask_trace, 3] = (m10[mask_trace] - m01[mask_trace]) / t

    mask_m00 = (~mask_trace) & (m00 > m11) & (m00 > m22)
    if mask_m00.any():
        t = torch.sqrt((1.0 + m00[mask_m00] - m11[mask_m00] - m22[mask_m00]).clamp_min(eps)) * 2.0
        q[mask_m00, 0] = (m21[mask_m00] - m12[mask_m00]) / t
        q[mask_m00, 1] = 0.25 * t
        q[mask_m00, 2] = (m01[mask_m00] + m10[mask_m00]) / t
        q[mask_m00, 3] = (m02[mask_m00] + m20[mask_m00]) / t

    mask_m11 = (~mask_trace) & (~mask_m00) & (m11 > m22)
    if mask_m11.any():
        t = torch.sqrt((1.0 + m11[mask_m11] - m00[mask_m11] - m22[mask_m11]).clamp_min(eps)) * 2.0
        q[mask_m11, 0] = (m02[mask_m11] - m20[mask_m11]) / t
        q[mask_m11, 1] = (m01[mask_m11] + m10[mask_m11]) / t
        q[mask_m11, 2] = 0.25 * t
        q[mask_m11, 3] = (m12[mask_m11] + m21[mask_m11]) / t

    mask_m22 = (~mask_trace) & (~mask_m00) & (~mask_m11)
    if mask_m22.any():
        t = torch.sqrt((1.0 + m22[mask_m22] - m00[mask_m22] - m11[mask_m22]).clamp_min(eps)) * 2.0
        q[mask_m22, 0] = (m10[mask_m22] - m01[mask_m22]) / t
        q[mask_m22, 1] = (m02[mask_m22] + m20[mask_m22]) / t
        q[mask_m22, 2] = (m12[mask_m22] + m21[mask_m22]) / t
        q[mask_m22, 3] = 0.25 * t

    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return q


def multivector_to_rotation_matrix(mv: torch.Tensor) -> torch.Tensor:
    if mv.ndim != 2 or mv.shape[-1] != 8:
        raise ValueError(f"Expected multivector shape [B, 8], got {tuple(mv.shape)}")

    v0 = mv[:, 0]
    v4 = mv[:, 4]
    v5 = mv[:, 5]
    v6 = mv[:, 6]

    norm = torch.sqrt((v0 * v0 + v4 * v4 + v5 * v5 + v6 * v6).clamp_min(1e-8))
    r0 = v0 / norm
    r4 = v4 / norm
    r5 = v5 / norm
    r6 = v6 / norm

    m = torch.zeros((mv.shape[0], 3, 3), dtype=mv.dtype, device=mv.device)
    m[:, 0, 0] = 1.0 - 2.0 * (r4 * r4 + r5 * r5)
    m[:, 0, 1] = 2.0 * (r0 * r4 - r5 * r6)
    m[:, 0, 2] = 2.0 * (r0 * r5 + r4 * r6)
    m[:, 1, 0] = -2.0 * (r0 * r4 + r5 * r6)
    m[:, 1, 1] = 1.0 - 2.0 * (r4 * r4 + r6 * r6)
    m[:, 1, 2] = 2.0 * (r0 * r6 - r4 * r5)
    m[:, 2, 0] = 2.0 * (r4 * r6 - r0 * r5)
    m[:, 2, 1] = 2.0 * (r0 * r6 + r4 * r5)
    m[:, 2, 2] = 1.0 - 2.0 * (r6 * r6 + r5 * r5)
    return m
