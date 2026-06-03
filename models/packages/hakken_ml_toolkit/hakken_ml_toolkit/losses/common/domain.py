from typing import Annotated

import torch
from torch import dtype as torchdtype

FloatTensorScalar = Annotated[torch.Tensor, "Represents a scalar float tensor"]


FloatTensor1D = Annotated[torch.Tensor, "Represents a float tensor with 1 dimension"]

FloatTensor2D = Annotated[torch.Tensor, "Represents a float tensor with 2 dimensions"]

LongTensor1D = Annotated[torch.Tensor, "Represents a long tensor with 1 dimension"]

LongTensor2D = Annotated[torch.Tensor, "Represents a long tensor with 2 dimensions"]

LongTensor3D = Annotated[torch.Tensor, "Represents a long tensor with 3 dimensions"]

TensorND = Annotated[torch.Tensor, "Represents a  tensor with more than 1 dimensions"]


def is_float_tensor_with_dim(
    tensor: torch.Tensor, dim: int, dtype: torchdtype = torch.float32
) -> bool:
    return isinstance(tensor, torch.Tensor) and tensor.dtype == dtype and tensor.dim() == dim


def is_long_tensor_with_dim(tensor: torch.Tensor, dim: int) -> bool:
    return isinstance(tensor, torch.Tensor) and tensor.dtype == torch.long and tensor.dim() == dim
