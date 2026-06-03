from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

from hakken_ml_toolkit.ml_base_structures.common.exceptions import (
    EntityNotInTripleError,
)
from hakken_ml_toolkit.ml_utils.extras import TensorCreator

if TYPE_CHECKING:
    import torch

    from hakken_ml_toolkit.ml_base_structures.common.entities import LongTensor2D


FactIndex: TypeAlias = tuple[int, int, int]
FactIndexList: TypeAlias = list[FactIndex]
Fact: TypeAlias = tuple[str, str, str]


def assert_is_fact_index(fact: Any) -> None:
    assert isinstance(fact, tuple)
    assert len(fact) == 3
    assert all(isinstance(e, int) for e in fact), f"fact: {fact}"


def assert_is_fact_index_list(fact_list: Any, length: int | None = None) -> None:
    assert isinstance(fact_list, list)
    if length is not None:
        assert len(fact_list) == length
    for fact in fact_list:
        assert_is_fact_index(fact)


class FactUtils(Protocol):
    @staticmethod
    def create(subject: str, relation: str, object: str) -> Fact:
        return (subject, relation, object)


class FactIndexUtils(Protocol):
    @staticmethod
    def hash(fact: FactIndex):
        return hash(fact)

    @staticmethod
    def equal(fact_1: FactIndex, fact_2: FactIndex) -> bool:
        return fact_1 == fact_2

    @staticmethod
    def has_entity(fact: FactIndex, entity: int) -> bool:
        return entity in (fact[0], fact[2])  # subject or object

    @staticmethod
    def other_entity(fact: FactIndex, entity: int) -> int:
        subject, _, obj = fact
        if entity == subject:
            return obj
        if entity == obj:
            return subject
        raise EntityNotInTripleError(entity)

    @staticmethod
    def to_tensor(
        facts: FactIndex | list[FactIndex], device: str | torch.device = "cpu"
    ) -> LongTensor2D:
        facts_list = [facts] if isinstance(facts, tuple) else facts

        return TensorCreator.long_tensor(facts_list, device=device)
