from typing import Any, Protocol, runtime_checkable

import torch

SelectionLike = slice | list[int] | list[bool] | torch.Tensor


@runtime_checkable
class BatchSelectorLike(Protocol):
    def __call__(self, **kwargs: Any) -> SelectionLike: ...


class RelationTypeSelector:
    def __init__(self, relation_type: int) -> None:
        self.relation_type = relation_type

    def __call__(self, **kwargs: Any):
        relation_types = kwargs["relation_types"]
        return relation_types == self.relation_type
