from typing import Annotated

import torch

FloatTensor2D = Annotated[torch.Tensor, "Represents a float tensor with 2 dimensions"]

LongTensor2D = Annotated[torch.Tensor, "Represents a long tensor with 2 dimensions"]
