import torch
from torch import dtype as torchdtype


def is_float_tensor_with_dim(
    tensor: torch.Tensor, dim: int, dtype: torchdtype = torch.float32
) -> bool:
    return isinstance(tensor, torch.Tensor) and tensor.dtype == dtype and tensor.dim() == dim


def is_long_tensor_with_dim(tensor: torch.Tensor, dim: int) -> bool:
    return isinstance(tensor, torch.Tensor) and tensor.dtype == torch.long and tensor.dim() == dim
