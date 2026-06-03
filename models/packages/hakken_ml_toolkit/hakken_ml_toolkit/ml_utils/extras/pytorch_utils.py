from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from lightning.pytorch import seed_everything
from torch import nn

from hakken_ml_toolkit.ml_utils.constants import ActivationType


class SinActivation(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super().__init__()  # init the base class

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the function.
        """
        return torch.sin(input)  # simply apply already implemented SiLU


class TensorCreator(Protocol):
    @staticmethod
    def long_tensor(
        values_list: list | np.ndarray, device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        return torch.tensor(values_list, dtype=torch.long, device=device)

    @staticmethod
    def float_tensor(
        values_list: list | np.ndarray,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor:
        return torch.tensor(values_list, dtype=dtype, device=device)

    @staticmethod
    def long_arange(num_elements: int, device: str | torch.device) -> torch.Tensor:
        return torch.arange(num_elements, dtype=torch.long, device=device)


class PyTorchUtils:
    @staticmethod
    def flush_gpu_memory() -> None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    @staticmethod
    def split_tensors(
        tensor: torch.Tensor,
        split_proportion: dict[str, float] | None = None,
        shuffle: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Splits a tensor into multiple parts based on specified proportions.

        Args:
            tensor: The tensor to split
            split_proportion: Dictionary mapping split names to proportions.
                Default is {"train": 0.9, "val": 0.05, "test": 0.05}
            shuffle: Whether to randomly shuffle the tensor before splitting.
                Default is False

        Returns:
            Dictionary mapping split names to corresponding tensor slices

        Raises:
            ValueError: If split proportions don't sum to approximately 1.0

        Note:
            The last split will contain any remaining elements to ensure
            all data is used even with rounding errors
        """

        if split_proportion is None:
            split_proportion = {"train": 0.9, "val": 0.05, "test": 0.05}
        total_proportion = sum(split_proportion.values())
        if not (0.999 <= total_proportion <= 1.001):
            msg = f"Split proportions must sum to 1.0, got {total_proportion}"
            raise ValueError(msg)

        num_elements = tensor.shape[0]

        if shuffle:
            indices = torch.randperm(num_elements)
            tensor = tensor[indices]

        tensors_dict = {}
        start_idx = 0
        for split_name, proportion in split_proportion.items():
            if split_name == list(split_proportion.keys())[-1]:
                end_idx = num_elements
            else:
                split_size = int(num_elements * proportion)
                end_idx = start_idx + split_size

            tensors_dict[split_name] = tensor[start_idx:end_idx]

            start_idx = end_idx

        return tensors_dict

    @staticmethod
    def fix_all_seeds(seed: int, device: str | torch.device = "cpu") -> None:
        seed_everything(seed)

        if torch.cuda.is_available() and (
            device != "cpu" or (isinstance(device, torch.device) and device.type != "cpu")
        ):
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # These settings can improve determinism on GPUs, but may reduce performance
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def activation(name: ActivationType, **kwargs) -> nn.Module:
        activation_fn: nn.Module | None = None
        if name == ActivationType.RELU:
            activation_fn = nn.ReLU(inplace=False)
        if name == ActivationType.TANH:
            activation_fn = nn.Tanh()
        if name == ActivationType.SELU:
            activation_fn = nn.SELU(inplace=False)
        if name == ActivationType.SIGMOID:
            activation_fn = nn.Sigmoid()
        if name == ActivationType.PRELU:
            activation_fn = nn.PReLU()
        if name == ActivationType.ELU:
            activation_fn = nn.ELU(inplace=False)
        if name == ActivationType.LEAKY_RELU:
            activation_fn = nn.LeakyReLU(**kwargs)
        if name == ActivationType.SOFTMAX:
            activation_fn = nn.Softmax()
        if name == ActivationType.IDENTITY:
            activation_fn = nn.Identity()
        if name == ActivationType.SIN:
            activation_fn = SinActivation()

        if activation_fn is None:
            raise NotImplementedError(f"Activation {name} not implemented.")
        return activation_fn

    @staticmethod
    def concat_tensors(tensors_list: list[torch.Tensor], dim: int = 1) -> torch.Tensor:
        return torch.cat(tensors_list, dim=dim)

    @staticmethod
    def tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
        """Detach tensor from computation graph and move to CPU."""
        return tensor.detach().cpu()

    @staticmethod
    def load(file_path: str | Path, weights_only: bool = False) -> torch.Tensor:
        return torch.load(file_path, weights_only=weights_only)  # type: ignore
