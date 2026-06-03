from typing import Annotated

import torch

LongTensor2D = Annotated[torch.Tensor, "Represents a long tensor with 2 dimensions"]


def is_long_tensor_with_dim(tensor: torch.Tensor, dim: int) -> bool:
    return isinstance(tensor, torch.Tensor) and tensor.dtype == torch.long and tensor.dim() == dim
