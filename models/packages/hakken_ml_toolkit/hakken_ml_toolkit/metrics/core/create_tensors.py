from typing import Protocol

import torch


class TensorCreator(Protocol):
    @staticmethod
    def long_tensor(values_list: list, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.tensor(values_list, dtype=torch.long, device=device)

    @staticmethod
    def float_tensor(
        values_list: list,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor:
        return torch.tensor(values_list, dtype=dtype, device=device)
