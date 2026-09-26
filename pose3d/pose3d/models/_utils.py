"""Small helpers shared by several model families."""

import torch
import torch.nn as nn


def so3_num_fourier_coeffs(lmax: int) -> int:
    return sum([(2 * l + 1) ** 2 for l in range(lmax + 1)])


def _move_tensors_in_object(obj, device: torch.device, visited: set):
    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)

    if isinstance(obj, torch.Tensor):
        return obj if obj.device == device else obj.to(device)

    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            obj[key] = _move_tensors_in_object(value, device, visited)
        return obj

    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            obj[idx] = _move_tensors_in_object(value, device, visited)
        return obj

    if isinstance(obj, tuple):
        return tuple(_move_tensors_in_object(value, device, visited) for value in obj)

    if hasattr(obj, "__dict__"):
        for attr_name, attr_value in list(vars(obj).items()):
            moved_value = _move_tensors_in_object(attr_value, device, visited)
            if moved_value is not attr_value:
                setattr(obj, attr_name, moved_value)
        return obj

    return obj


def move_unregistered_tensors_to_device(module: nn.Module, device: torch.device):
    visited = set()
    for submodule in module.modules():
        registered_params = set(dict(submodule.named_parameters(recurse=False)).keys())
        registered_buffers = set(dict(submodule.named_buffers(recurse=False)).keys())

        for attr_name, attr_value in list(submodule.__dict__.items()):
            if attr_name in registered_params or attr_name in registered_buffers:
                continue
            moved_value = _move_tensors_in_object(attr_value, device, visited)
            if moved_value is not attr_value:
                setattr(submodule, attr_name, moved_value)
