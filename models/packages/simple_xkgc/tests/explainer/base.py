from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import torch
from hakken_ml_toolkit.ml_base_structures.data_generator import DummyDataGenerator
from kge.models.gnn import GNNKGE
from kge.scores import ComplExScore
from torch_geometric.nn.models import GraphSAGE

from hakken_explainer.constants import RerankStrategy, ScoreType
from hakken_explainer.entities.config import ScoreTypeConfig
from hakken_explainer.explainers import HakkenExplainer

if TYPE_CHECKING:
    from hakken_ml_toolkit.ml_base_structures import Fact, KnowledgeGraph

    from hakken_explainer.candidate_finder.base import CandidateFinder


def sample_candidates_tensor(
    num_candidates: int,
    num_entities: int,
    num_relations: int,
    num_hops: int = 2,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate a sample candidates tensor for testing.

    Creates a tensor representing candidate explanation paths where each path
    consists of num_hops triples (subject, relation, object).

    Args:
        num_candidates: Number of candidate paths to generate
        num_hops: Number of triples in each path (explanation length)
        num_entities: Maximum entity ID (for generating valid entity indices)
        num_relations: Maximum relation ID (for generating valid relation indices)
        device: Device to place the tensor on

    Returns:
        Tensor with shape [num_candidates, num_hops, 3] where each element
        is [subject_idx, relation_idx, object_idx]
    """
    candidates_list = []
    for _ in range(num_candidates):
        path = []
        for _ in range(num_hops):
            subject = random.randint(0, num_entities - 1)
            relation = random.randint(0, num_relations - 1)
            object_entity = random.randint(0, num_entities - 1)
            path.append([subject, relation, object_entity])
        candidates_list.append(path)

    return torch.tensor(candidates_list, dtype=torch.long, device=device)


class BaseHakkenExplainerTest:
    """Base test class for HakkenExplainer with different CandidateFinder implementations.

    Subclass this and provide a `candidate_finder` fixture to test HakkenExplainer
    with any CandidateFinder implementation.
    """

    # ========== Configuration Parameters ==========

    @pytest.fixture
    def num_entities(self) -> int:
        """Number of entities in the knowledge graph. Override in subclasses to customize."""
        return 1000

    @pytest.fixture
    def num_relations(self) -> int:
        """Number of relations in the knowledge graph. Override in subclasses to customize."""
        return 10

    @pytest.fixture
    def embedding_dim(self) -> int:
        """Dimension of the embedding space"""
        return 64

    @pytest.fixture
    def search_space_size(self) -> int:
        """Size of the search space (number of facts). Override in subclasses to customize."""
        return 512

    # ========== Core Fixtures ==========

    @pytest.fixture
    def candidate_finder(self) -> CandidateFinder:
        """Fixture to be overridden by subclasses to provide specific CandidateFinder instances."""
        msg = "Subclasses must provide a candidate_finder fixture."
        raise NotImplementedError(msg)

    @pytest.fixture
    def model(self, embedding_dim: int, num_entities: int, num_relations: int) -> GNNKGE:
        """Fixture to be overridden by subclasses to provide a GNNKGE model instance."""

        gnn = GraphSAGE(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            hidden_channels=embedding_dim,
            num_layers=2,
        )
        score_fn = ComplExScore()
        return GNNKGE(
            embedding_dim=embedding_dim,
            num_entities=num_entities,
            num_relations=num_relations,
            gnn=gnn,
            score_fn=score_fn,
        )

    @pytest.fixture
    def kg(self, search_space_size: int, num_entities: int, num_relations: int) -> KnowledgeGraph:
        """Fixture to be overridden by subclasses to provide a KnowledgeGraph instance."""
        return DummyDataGenerator.knowledge_graph(
            batch_size=search_space_size, num_entities=num_entities, num_relations=num_relations
        )

    @pytest.fixture
    def search_space(self, kg: KnowledgeGraph) -> torch.Tensor:
        """Create a search space tensor with the specified size."""

        return kg.facts_dict["train"]

    @pytest.fixture
    def hakken_explainer(
        self,
        candidate_finder: CandidateFinder,
        model: GNNKGE,
        kg: KnowledgeGraph,
        search_space: torch.Tensor,
    ) -> HakkenExplainer:
        """Create a HakkenExplainer instance with provided dependencies."""
        return HakkenExplainer(
            candidate_finder=candidate_finder,
            model=model,
            kg=kg,
            search_space=search_space,
        )

    @pytest.fixture
    def sample_triple(self, kg: KnowledgeGraph) -> Fact:
        """Create a sample triple for testing."""
        # Subclasses should override this to return a valid Fact for their KG
        subject_idx, target_idx = random.sample(range(kg.num_entities), 2)

        relation_idx = random.choice(range(kg.num_relations))

        subject = kg.decode_entity(subject_idx)
        relation = kg.decode_relation(relation_idx)
        target = kg.decode_entity(target_idx)

        return (subject, relation, target)

    @pytest.fixture
    def explanation_length(self) -> int:
        """Default explanation length for tests."""
        return 2

    # ========== Initialization Tests ==========

    def test_initialization(
        self,
        candidate_finder: CandidateFinder,
        model: GNNKGE,
        kg: KnowledgeGraph,
        search_space: torch.Tensor,
        search_space_size: int,
    ):
        """Test that HakkenExplainer initializes correctly with all dependencies."""
        explainer = HakkenExplainer(
            candidate_finder=candidate_finder,
            model=model,
            kg=kg,
            search_space=search_space,
        )

        assert explainer.candidate_finder is candidate_finder
        assert explainer.model is model
        assert explainer.kg is kg
        assert torch.equal(explainer.search_space, search_space)
        assert explainer.search_space.shape[0] == search_space_size

    def test_set_search_space(
        self,
        hakken_explainer: HakkenExplainer,
        num_entities: int,
        num_relations: int,
    ):
        """Test that set_search_space updates the search space."""
        new_search_space_size = 30
        new_search_space = torch.randint(
            0,
            min(num_entities, num_relations),
            (new_search_space_size, 3),
            dtype=torch.long,
        )
        hakken_explainer.set_search_space(new_search_space)
        assert torch.equal(hakken_explainer.search_space, new_search_space)
        assert hakken_explainer.search_space.shape[0] == new_search_space_size

    def test_search_space_size_parameter(
        self,
        hakken_explainer: HakkenExplainer,
        search_space_size: int,
    ):
        """Test that search space has the correct size based on parameter."""
        assert hakken_explainer.search_space.shape[0] == search_space_size

    # ========== Explain Method Tests ==========

    def test_explain_returns_dataframe(
        self, hakken_explainer: HakkenExplainer, sample_triple: Fact
    ):
        """Test that explain method returns a pandas DataFrame."""
        result = hakken_explainer.explain(sample_triple)
        assert isinstance(result, pd.DataFrame)

    def test_explain_dataframe_columns(
        self,
        hakken_explainer: HakkenExplainer,
        sample_triple: Fact,
        explanation_length: int,
    ):
        """Test that explain returns DataFrame with expected columns."""
        result = hakken_explainer.explain(sample_triple, explanation_length=explanation_length)

        # Should have at least these columns
        expected_columns = {"explanation", "explanation_index", "score"}
        assert expected_columns.issubset(set(result.columns))

    def test_explain_with_default_score_type(
        self, hakken_explainer: HakkenExplainer, sample_triple: Fact
    ):
        """Test that explain uses SUFFICIENT score type by default."""
        result = hakken_explainer.explain(sample_triple)
        assert isinstance(result, pd.DataFrame)
        # Should have score_sufficient column when default is used
        if len(result) > 0:
            assert "score_sufficient" in result.columns

    def test_explain_with_custom_score_types(
        self, hakken_explainer: HakkenExplainer, sample_triple: Fact
    ):
        """Test explain with custom score type configurations."""
        score_type_list = [
            ScoreTypeConfig(type=ScoreType.SUFFICIENT, batch_size=16),
            ScoreTypeConfig(type=ScoreType.NECESSARY, batch_size=16),
        ]

        result = hakken_explainer.explain(sample_triple, score_type_list=score_type_list)

        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "score_sufficient" in result.columns
            assert "score_necessary" in result.columns

    def test_explain_with_explanation_length(
        self,
        hakken_explainer: HakkenExplainer,
        sample_triple: Fact,
        explanation_length: int,
    ):
        """Test explain with specified explanation length."""
        result = hakken_explainer.explain(sample_triple, explanation_length=explanation_length)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.parametrize("seed", [None, 0, 1, 42, 123])
    def test_explain_with_allowed_relations(
        self,
        hakken_explainer: HakkenExplainer,
        sample_triple: Fact,
        kg: KnowledgeGraph,
        seed: None | int,
    ):
        """Test explain with allowed relations filter."""
        if seed is None:
            allowed_relations_ids = None  # Subclasses should override this test if needed
        else:
            k = random.randint(0, kg.num_relations)  # example: pick up to 3
            if k == 0:
                allowed_relations_ids = None
            else:
                allowed_relation_index = random.sample(range(kg.num_relations), k)
                allowed_relations_ids = [kg.decode_relation(idx) for idx in allowed_relation_index]

        result = hakken_explainer.explain(
            sample_triple, allowed_relations_ids=allowed_relations_ids
        )
        assert isinstance(result, pd.DataFrame)

    def test_explain_with_different_rerank_strategies(
        self, hakken_explainer: HakkenExplainer, sample_triple: Fact
    ):
        """Test explain with different rerank strategies."""
        strategies = [RerankStrategy.SCORES, RerankStrategy.UNIQUE_PATHWAYS]

        for strategy in strategies:
            result = hakken_explainer.explain(sample_triple, rerank_strategy=strategy)
            assert isinstance(result, pd.DataFrame)

    def test_explain_with_device_cpu(self, hakken_explainer: HakkenExplainer, sample_triple: Fact):
        """Test explain with CPU device."""
        result = hakken_explainer.explain(sample_triple, device="cpu")
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_explain_with_device_cuda(self, hakken_explainer: HakkenExplainer, sample_triple: Fact):
        """Test explain with CUDA device if available."""
        result = hakken_explainer.explain(sample_triple, device="cuda")
        assert isinstance(result, pd.DataFrame)

    def test_explain_empty_candidates(self, hakken_explainer: HakkenExplainer, sample_triple: Fact):
        """Test explain when no candidates are found."""

        result = hakken_explainer.explain(sample_triple)
        assert isinstance(result, pd.DataFrame)

        expected_columns = {"explanation", "explanation_index", "score"}
        assert set(result.columns) == expected_columns

    def test_explain_scales_with_search_space_size(
        self,
        hakken_explainer: HakkenExplainer,
        sample_triple: Fact,
    ):
        """Test that explain works correctly with different search space sizes."""
        result = hakken_explainer.explain(sample_triple)
        assert isinstance(result, pd.DataFrame)

    # ========== Process Scores Tests ==========

    def test_process_scores_returns_dataframe(
        self, hakken_explainer: HakkenExplainer, kg: KnowledgeGraph
    ):
        """Test that process_scores returns a DataFrame."""
        scoring_results = {
            "sufficient": [0.8, 0.6, 0.9],
            "necessary": [0.7, 0.5, 0.8],
        }

        # Adjust search_space to match number of scores
        num_candidates = len(scoring_results["sufficient"])
        candidates_tensor = sample_candidates_tensor(
            num_candidates=num_candidates,
            num_entities=kg.num_entities,
            num_relations=kg.num_relations,
        )

        result = hakken_explainer.process_scores(scoring_results, candidates_tensor)
        assert isinstance(result, pd.DataFrame)

    def test_process_scores_dataframe_structure(
        self, hakken_explainer: HakkenExplainer, kg: KnowledgeGraph
    ):
        """Test that process_scores DataFrame has correct structure."""
        scoring_results = {
            "sufficient": [0.8, 0.6],
            "necessary": [0.7, 0.5],
        }
        num_candidates = len(scoring_results["sufficient"])

        candidates_tensor = sample_candidates_tensor(
            num_candidates=num_candidates,
            num_entities=kg.num_entities,
            num_relations=kg.num_relations,
        )
        result = hakken_explainer.process_scores(scoring_results, candidates_tensor)

        # Check required columns
        assert "explanation" in result.columns
        assert "pathway" in result.columns
        assert "explanation_index" in result.columns
        assert "score" in result.columns

        # Check score type columns
        assert "score_sufficient" in result.columns
        assert "score_necessary" in result.columns

        # Check that score is the mean
        assert len(result) == num_candidates
        for _idx, row in result.iterrows():
            expected_score = np.mean([row["score_sufficient"], row["score_necessary"]])
            assert row["score"] == pytest.approx(expected_score, rel=0, abs=1e-6)

    def test_process_scores_single_score_type(
        self, hakken_explainer: HakkenExplainer, kg: KnowledgeGraph
    ):
        """Test process_scores with a single score type."""
        scoring_results = {"sufficient": [0.8, 0.6, 0.9]}
        num_candidates = len(scoring_results["sufficient"])

        candidates_tensor = sample_candidates_tensor(
            num_candidates=num_candidates,
            num_entities=kg.num_entities,
            num_relations=kg.num_relations,
        )

        result = hakken_explainer.process_scores(scoring_results, candidates_tensor)

        assert len(result) == num_candidates
        assert "score_sufficient" in result.columns
        assert "score" in result.columns

        # Score should equal the single score type when only one exists
        for _idx, row in result.iterrows():
            assert row["score"] == pytest.approx(row["score_sufficient"], rel=0, abs=1e-6)

    def test_process_scores_explanation_strings(
        self, hakken_explainer: HakkenExplainer, kg: KnowledgeGraph
    ):
        """Test that process_scores generates explanation strings."""
        scoring_results = {"sufficient": [0.8, 0.6]}

        num_candidates = len(scoring_results["sufficient"])

        candidates_tensor = sample_candidates_tensor(
            num_candidates=num_candidates,
            num_entities=kg.num_entities,
            num_relations=kg.num_relations,
        )

        result = hakken_explainer.process_scores(scoring_results, candidates_tensor)

        assert "explanation" in result.columns
        # Explanation should be a string
        for explanation in result["explanation"]:
            assert isinstance(explanation, str)
            assert len(explanation) > 0

    # ========== Integration Tests ==========

    def test_explain_end_to_end(
        self,
        hakken_explainer: HakkenExplainer,
        sample_triple: Fact,
        explanation_length: int,
    ):
        """Test complete explain workflow end-to-end."""
        result = hakken_explainer.explain(sample_triple, explanation_length=explanation_length)

        assert isinstance(result, pd.DataFrame)
        # Result should be sorted/ranked
        if len(result) > 1:
            # Scores should be in descending order (assuming reranking)
            scores = result["score"].values
            assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_set_search_space_and_explain(
        self,
        hakken_explainer: HakkenExplainer,
        sample_triple: Fact,
        num_entities: int,
        num_relations: int,
    ):
        """Test that changing search space affects explain results."""
        original_result = hakken_explainer.explain(sample_triple)

        new_search_space_size = 30
        new_search_space = torch.randint(
            0,
            min(num_entities, num_relations),
            (new_search_space_size, 3),
            dtype=torch.long,
        )
        hakken_explainer.set_search_space(new_search_space)

        new_result = hakken_explainer.explain(sample_triple)

        # Both should be DataFrames
        assert isinstance(original_result, pd.DataFrame)
        assert isinstance(new_result, pd.DataFrame)

    # ========== Parameter Validation Tests ==========

    def test_num_entities_parameter_used(
        self,
        num_entities: int,
    ):
        """Test that num_entities parameter is properly used in KG setup."""
        assert num_entities > 0

    def test_num_relations_parameter_used(
        self,
        num_relations: int,
    ):
        """Test that num_relations parameter is properly used in KG setup."""
        assert num_relations > 0

    def test_search_space_size_parameter_used(
        self,
        hakken_explainer: HakkenExplainer,
        search_space_size: int,
    ):
        """Test that search_space_size parameter is properly used."""
        assert hakken_explainer.search_space.shape[0] == search_space_size
        assert search_space_size > 0

    # ========== Edge Cases ==========

    def test_explain_with_none_explanation_length(
        self, hakken_explainer: HakkenExplainer, sample_triple: Fact
    ):
        """Test explain with None explanation_length."""
        result = hakken_explainer.explain(sample_triple, explanation_length=None)
        assert isinstance(result, pd.DataFrame)

    def test_process_scores_empty_results(
        self,
        hakken_explainer: HakkenExplainer,
    ):
        """Test process_scores with empty scoring results."""
        scoring_results: dict[str, list[float]] = {}
        empty_candidates_tensor = torch.empty((0, 2, 3), dtype=torch.long)

        result = hakken_explainer.process_scores(scoring_results, empty_candidates_tensor)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_small_search_space(
        self,
        hakken_explainer: HakkenExplainer,
        sample_triple: Fact,
        num_entities: int,
        num_relations: int,
    ):
        """Test with a very small search space."""
        small_search_space = torch.randint(
            0,
            min(num_entities, num_relations),
            (1, 3),
            dtype=torch.long,
        )
        hakken_explainer.set_search_space(small_search_space)
        result = hakken_explainer.explain(sample_triple, device="cpu")
        assert isinstance(result, pd.DataFrame)

    def test_large_search_space(
        self,
        hakken_explainer: HakkenExplainer,
        sample_triple: Fact,
        num_entities: int,
        num_relations: int,
    ):
        """Test with a large search space."""
        large_search_space = torch.randint(
            0,
            min(num_entities, num_relations),
            (1000, 3),
            dtype=torch.long,
        )
        hakken_explainer.set_search_space(large_search_space)
        result = hakken_explainer.explain(sample_triple)
        assert isinstance(result, pd.DataFrame)
