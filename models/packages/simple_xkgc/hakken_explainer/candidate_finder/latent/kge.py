from typing import Unpack

from hakken_ml_toolkit.ml_base_structures.fact import FactIndexList
from hakken_ml_toolkit.ml_base_structures.knowledge_graph import KnowledgeGraph
from kge.models.base import KGEI
from loguru import logger

from hakken_explainer.candidate_finder.base import CandidateFinder, SetupKwargsBase
from hakken_explainer.candidate_finder.path_generator import KGEPathGenerator
from hakken_explainer.exceptions import MissingRequiredArgumentError

DEFAULT_MIN_SCORE = 0.5


class SetupKwargsKGE(SetupKwargsBase, total=False):
    """Extended kwargs for RandomCandidateFinder setup."""

    kg: KnowledgeGraph
    kge: KGEI


class KGEPathCandidateFinder(CandidateFinder):
    def __init__(self, min_score: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.min_score = min_score

    def setup(self, **kwargs: Unpack[SetupKwargsKGE]) -> None:
        logger.warning("Not calling super().setup()")
        kg: KnowledgeGraph | None = kwargs.get("kg")
        if kg is None:
            raise MissingRequiredArgumentError(argument_name="kg")

        model: KGEI | None = kwargs.get("kge")
        if model is None:
            raise MissingRequiredArgumentError(argument_name="model")

        self.path_gen = KGEPathGenerator(
            entity_indices=kg.get_entity_indices(),
            relation_indices=kg.get_relation_indices(),
            model=model,
            min_score=self.min_score,
        )
        self.path_gen.to_device(self.device)

    def find_candidates(
        self,
        source: int,
        target: int,
        relation: int | None = None,
        k: int | None = None,
        allowed_relations: list[int] | None = None,
    ) -> list[FactIndexList]:
        if relation is not None:
            logger.warning("relation will be ignored")

        if k is None:
            logger.warning("k is None. Setting k=2.")
            k = 2
        # TODO: Remove duplicate paths
        return self.path_gen.generate_paths(
            source=source,
            target=target,
            num_hops=k,
            allowed_relations=allowed_relations,
            num_paths=self.max_candidates,
        )
