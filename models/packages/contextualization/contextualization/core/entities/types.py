from typing import Annotated, TypeAlias

import ml_dtypes
import numpy as np
import torch
from numpy import typing as npt
from pydantic import StringConstraints

StripString: TypeAlias = Annotated[str, StringConstraints(strip_whitespace=True)]

NumpyVector: TypeAlias = npt.NDArray[np.float32 | np.float16 | ml_dtypes.bfloat16]
TorchVector: TypeAlias = torch.Tensor
Vector: TypeAlias = list[float] | NumpyVector | TorchVector
