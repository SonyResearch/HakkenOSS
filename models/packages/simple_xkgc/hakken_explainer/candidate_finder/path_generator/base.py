from abc import ABC, abstractmethod

import numpy as np
import torch
from hakken_ml_toolkit.ml_base_structures.fact import FactIndex, FactIndexList


class PathGenerator(ABC):
    """Abstract base class for generating facts and paths in knowledge graphs.

    This class provides a framework for generating facts (subject, relation, object)
    and paths of facts using various generative models. It supports device management
    for GPU/CPU computation and includes utility methods for data conversion.
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = device

    def to_device(self, device: str | torch.device = "cpu") -> None:
        self.device = device

    def convert_to_numpy(self, values: list[int] | torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert input values to a NumPy array.

        Supports conversion from Python lists, PyTorch tensors, or NumPy arrays.
        PyTorch tensors are moved to CPU before conversion.

        Args:
            values: Input values as a list of integers, PyTorch tensor, or NumPy array.

        Returns:
            NumPy array representation of the input values.

        Raises:
            TypeError: If the input type is not supported.
        """

        if isinstance(values, np.ndarray):
            return values
        if isinstance(values, torch.Tensor):
            return values.cpu().numpy()
        if isinstance(values, list):
            return np.array(values)

        msg = f"Unsupported type: {type(values)}. Expected list[int], torch.Tensor, or np.ndarray."
        raise TypeError(msg)

    def convert_to_torch_tensor(
        self, values: list[int] | torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        """Convert input values to a PyTorch tensor on the specified device.

        Args:
            values: Input values as a list of integers, PyTorch tensor, or NumPy array.

        Returns:
            PyTorch tensor on the specified device.
        """
        if isinstance(values, torch.Tensor):
            return values.to(self.device)
        if isinstance(values, np.ndarray):
            return torch.from_numpy(values).to(self.device)
        if isinstance(values, list):
            return torch.tensor(values, dtype=torch.long, device=self.device)

        msg = f"Unsupported type: {type(values)}. Expected list[int], torch.Tensor, or np.ndarray."
        raise TypeError(msg)

    @abstractmethod
    def generate_facts(
        self,
        source: list[int] | torch.Tensor | None = None,
        allowed_relations: list[int] | None = None,
        target: list[int] | torch.Tensor | None = None,
        num_facts_per_entity: int = 1,
    ) -> list[FactIndex]:
        """Generate facts (subject, relation, object) for entities in a knowledge graph.

        Args:
            source: Source entity indices. If None, facts are generated without
                source constraints. Defaults to None.
            allowed_relations: Relation indices that can be used in generated facts.
                If None, all relations are allowed. Defaults to None.
            target: Target entity indices. If None, facts are generated without
                target constraints. Defaults to None.
            num_facts_per_entity: Number of facts to generate per entity. Defaults to 1.

        Returns:
            List of generated facts, each represented as a tuple (subject, relation, object).
        """
        pass

    @abstractmethod
    def generate_paths(
        self,
        source: int,
        target: int,
        num_hops: int,
        previous_generated_paths: list[FactIndexList] | None = None,
        allowed_relations: list[int] | None = None,
        num_paths: int = 1,
    ) -> list[FactIndexList]:
        """Generate paths of facts connecting a source entity to a target entity.

        A path consists of a sequence of facts where each fact connects entities,
        forming a chain from source to target with the specified number of hops.

        Args:
            source: The starting entity index.
            target: The target entity index.
            num_hops: The number of hops (intermediate facts) in the path.
            previous_generated_paths: Previously generated paths to consider or avoid
                during generation. Defaults to None.
            allowed_relations: Relation indices that can be used in path generation.
                If None, all relations are allowed. Defaults to None.
            num_paths: Number of distinct paths to generate. Defaults to 1.

        Returns:
            List of generated paths, where each path is a list of facts represented
            as tuples (subject, relation, object).
        """
        pass
