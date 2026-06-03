from dataclasses import dataclass
from typing import Annotated

import torch

FloatTensor1D = Annotated[torch.Tensor, "Represents a float tensor with 1 dimension"]
FloatTensor2D = Annotated[torch.Tensor, "Represents a float tensor with 2 dimensions"]

LongTensor1D = Annotated[torch.Tensor, "Represents a long tensor with 1 dimension"]

LongTensor2D = Annotated[torch.Tensor, "Represents a long tensor with 2 dimensions"]

LongTensor3D = Annotated[torch.Tensor, "Represents a long tensor with 3 dimensions"]

TensorND = Annotated[torch.Tensor, "Represents a  tensor with more than 1 dimensions"]


@dataclass
class ProximityNetworkData:
    neighbors: torch.Tensor  # Padded tensor of shape (num_entities, max_neighbors)
    distances: torch.Tensor  # Padded tensor of shape (num_entities, max_neighbors)
    neighbor_counts: torch.Tensor  # Tensor of shape (num_entities,)
