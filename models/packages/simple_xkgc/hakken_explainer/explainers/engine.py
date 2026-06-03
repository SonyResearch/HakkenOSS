from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from hakken_ml_toolkit.ml_utils.extras import TensorCreator
from loguru import logger

from hakken_explainer.constants import RerankStrategy, ScoreType
from hakken_explainer.entities.config import ScoreTypeConfig
from hakken_explainer.reranker.builder import RerankerBuilder
from hakken_explainer.scores.builder import ExplinerScoreBuilder
from hakken_explainer.utils import ExplainerUtils

if TYPE_CHECKING:
    import torch
    from hakken_ml_toolkit.ml_base_structures import Fact, KnowledgeGraph
    from kge.models.gnn import GNNKGE
    from torch import Tensor

    from hakken_explainer.candidate_finder.base import CandidateFinder


class HakkenExplainer:
    def __init__(
        self,
        candidate_finder: CandidateFinder,
        model: GNNKGE,
        kg: KnowledgeGraph,
        search_space: torch.Tensor,
    ) -> None:
        self.candidate_finder = candidate_finder
        self.model = model
        self.kg = kg

        self.search_space = search_space

    def set_search_space(self, search_space: torch.Tensor) -> None:
        self.search_space = search_space

    def explain(
        self,
        triple_to_probe: Fact,
        device: str | torch.device = "cuda",
        explanation_length: int | None = None,
        score_type_list: list[ScoreTypeConfig] | None = None,
        rerank_strategy: RerankStrategy = RerankStrategy.SCORES,
        allowed_relations_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate explanations for a knowledge graph triple prediction.

        Finds paths between subject and object entities, scores them using specified
        explanation types (sufficient/necessary), and returns ranked explanations
        with metadata and scores.

        Args:
            triple_to_probe: Target triple to explain
            batch_size: Batch size for scoring computations
            device: Device for tensor operations
            min_length: Minimum path length (overrides shortest path if longer)
            score_type_list: Types of explanations to compute (defaults to SUFFICIENT)
            rerank_strategy: Strategy for ranking final results
            allowed_relations_ids: Relations that are allowed

        Returns:
            DataFrame with explanation strings, scores, and metadata, sorted by relevance
        """
        if score_type_list is None:
            score_type_list = [ScoreTypeConfig(type=ScoreType.SUFFICIENT, batch_size=32)]

        logger.info(f"Starting Explanation computation for {triple_to_probe}")

        logger.info(f"Looking for candidates with length {explanation_length}")
        subject_idx = self.kg.encode_entity(triple_to_probe[0])
        relation_idx = self.kg.encode_relation(triple_to_probe[1])
        object_idx = self.kg.encode_entity(triple_to_probe[2])

        allowed_relations: list[int] | None = None

        if allowed_relations_ids is not None:
            allowed_relations = self.kg.encode_relations(allowed_relations_ids)

        candidates_list = self.candidate_finder.find_candidates(
            source=subject_idx,
            target=object_idx,
            relation=relation_idx,
            k=explanation_length,
            allowed_relations=allowed_relations,
        )

        logger.info(f"Found {len(candidates_list)} candidate explanations")
        if len(candidates_list) == 0:
            return pd.DataFrame(columns=["explanation", "explanation_index", "score"])

        logger.info("Converting candidates to tensor")

        candidates_tensor = TensorCreator.long_tensor(candidates_list, device=device)
        logger.info(f"candidates_tensor: {candidates_tensor.shape}")

        sro_to_probe = self.kg.encode_facts_as_tensor([triple_to_probe])
        sro_to_probe = sro_to_probe.to(device)

        scoring_results = {}

        for expl_config in score_type_list:
            scorer = ExplinerScoreBuilder.from_kwargs(
                expl_config.type, context_kg=self.search_space, model=self.model
            )
            scores = scorer.score(
                target_fact=sro_to_probe,
                candidate_paths=candidates_tensor,
                batch_size=expl_config.batch_size,
                num_hops=2,
                normalize_by_original=True,
            )
            scoring_results[expl_config.type.value] = scores

        df_expl = self.process_scores(scoring_results, candidates_tensor)

        reranker = RerankerBuilder.build(rerank_strategy)

        return reranker.rerank(df_expl)

    def process_scores(
        self, scoring_results: dict[str, list[float]], candidates_tensor: Tensor
    ) -> pd.DataFrame:
        """Convert explanation scores into a structured DataFrame with metadata.

        Transforms raw scoring results into a comprehensive DataFrame containing
        explanation strings, pathways, indices, individual scores by type, and
        averaged scores for analysis and ranking.

        Args:
            scoring_results: Dictionary mapping score type names to lists of scores
                for each explanation (e.g., {'necessary': [0.8, 0.6], 'sufficient': [0.7, 0.9]})
            candidates_tensor: Tensor of explanation paths with shape [num_expl, path_length, 3]

        Returns:
            DataFrame with columns: explanation, pathway, explanation_index,
            score_{type} for each scoring type, and averaged 'score' column
        """
        data_list: list[dict] = []
        for i, explanation_index_i in enumerate(candidates_tensor):
            explanation_id_i = self.kg.decode_facts(explanation_index_i)
            pathway_id_i = [[s, o] for s, _r, o in explanation_id_i]

            explanation_str = ExplainerUtils.convert_triple_path_id_to_str(explanation_id_i)
            data_i = {
                "explanation": explanation_str,
                "pathway": str(pathway_id_i),
                "explanation_index": explanation_index_i.cpu().tolist(),
            }
            scores_i = []

            for key, values in scoring_results.items():
                value = values[i]

                scores_i.append(value)
                data_i[f"score_{key}"] = value

            data_i["score"] = np.mean(scores_i)

            data_list.append(data_i)

        return pd.DataFrame(data_list)
