from typing import Unpack

from hakken_ml_toolkit.ml_base_structures.fact import FactIndexList
from hakken_ml_toolkit.ml_base_structures.knowledge_graph import KnowledgeGraph
from loguru import logger

from hakken_explainer.candidate_finder.base import CandidateFinder, SetupKwargsBase
from hakken_explainer.candidate_finder.path_generator.random import RandomPathGenerator
from hakken_explainer.exceptions import MissingRequiredArgumentError


class SetupKwargsRandom(SetupKwargsBase, total=False):
    """Extended kwargs for RandomCandidateFinder setup."""

    kg: KnowledgeGraph


class RandomPathCandidateFinder(CandidateFinder):
    def setup(self, **kwargs: Unpack[SetupKwargsRandom]) -> None:
        """Build NetworkX graph from facts_batch."""
        logger.warning("Not calling super().setup()")

        kg: KnowledgeGraph | None = kwargs.get("kg")
        if kg is None:
            raise MissingRequiredArgumentError(argument_name="kg")

        self.kg = kg

        self.path_gen = RandomPathGenerator(
            entity_indices=self.kg.get_entity_indices(),
            relation_indices=self.kg.get_relation_indices(),
        )

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
        # TODO: Remove duplicate paths
        if k is None:
            logger.warning("k is None. Setting k=2.")
            k = 2
        return self.path_gen.generate_paths(
            source=source,
            target=target,
            num_hops=k,
            allowed_relations=allowed_relations,
            num_paths=self.max_candidates,
        )
