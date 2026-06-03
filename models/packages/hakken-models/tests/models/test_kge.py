from typing import cast

import pytest
import torch

from hakken_models.models.kge import KGE
from hakken_models.scores import score_fn_registry
from hakken_models.scores.base import ScoreFn

# ============================================================================
# Test Configuration
# ============================================================================
KGE_CONFIGS = [
    {"num_entities": 1_000, "num_relations": 10, "embedding_dim": 32},
    {"num_entities": 10_000, "num_relations": 20, "embedding_dim": 64},
    {"num_entities": 100, "num_relations": 5, "embedding_dim": 16},
]

SCORE_FN_NAMES = score_fn_registry.list_all()


# ============================================================================
# Test Class
# ============================================================================
class TestKGE:
    """Test KGE model with different configurations and score functions."""

    @pytest.fixture(params=KGE_CONFIGS)
    def kge_config(self, request: pytest.FixtureRequest) -> dict[str, int]:
        """Parametrized KGE configuration."""
        return cast(dict, request.param)

    @pytest.fixture(params=SCORE_FN_NAMES)
    def score_fn_name(self, request: pytest.FixtureRequest) -> str:
        """Parametrized score function name."""
        return cast(str, request.param)

    @pytest.fixture
    def score_fn(self, score_fn_name: str, kge_config: dict[str, int]) -> ScoreFn:
        """Score function instance from registry."""
        from hakken_models.scores.conv_kb import ConvKBScore

        cls = score_fn_registry.get(score_fn_name)
        if issubclass(cls, ConvKBScore):
            return score_fn_registry.create(score_fn_name, emb_dim=kge_config["embedding_dim"])
        return score_fn_registry.create(score_fn_name)

    @pytest.fixture
    def kge(self, kge_config: dict[str, int], score_fn: ScoreFn) -> KGE:
        """KGE model instance."""
        return KGE(
            num_entities=kge_config["num_entities"],
            num_relations=kge_config["num_relations"],
            embedding_dim=kge_config["embedding_dim"],
            score_fn=score_fn,
        )

    @pytest.fixture
    def sample_facts(self, kge_config: dict[str, int]) -> torch.Tensor:
        """Sample facts tensor for testing."""
        num_entities = kge_config["num_entities"]
        num_relations = kge_config["num_relations"]
        return torch.tensor(
            [
                [0, 0, 1],
                [
                    min(1, num_entities - 1),
                    min(1, num_relations - 1),
                    min(2, num_entities - 1),
                ],
                [
                    min(2, num_entities - 1),
                    min(2, num_relations - 1),
                    min(3, num_entities - 1),
                ],
            ],
            dtype=torch.long,
        )

    @pytest.fixture
    def large_facts(self, kge_config: dict[str, int]) -> torch.Tensor:
        """Larger facts tensor for testing."""
        num_entities = kge_config["num_entities"]
        num_relations = kge_config["num_relations"]
        batch_size = 20
        facts = torch.zeros(batch_size, 3, dtype=torch.long)
        facts[:, 0] = torch.randint(0, num_entities, (batch_size,))
        facts[:, 1] = torch.randint(0, num_relations, (batch_size,))
        facts[:, 2] = torch.randint(0, num_entities, (batch_size,))
        return facts

    @pytest.fixture
    def facts_with_extra_dims(self, kge_config: dict[str, int]) -> torch.Tensor:
        """Facts with extra dimensions (M > 3)."""
        num_entities = kge_config["num_entities"]
        num_relations = kge_config["num_relations"]
        batch_size = 10
        num_components = 5
        facts = torch.zeros(batch_size, num_components, dtype=torch.long)
        facts[:, 0] = torch.randint(0, num_entities, (batch_size,))
        facts[:, 1] = torch.randint(0, num_relations, (batch_size,))
        facts[:, 2] = torch.randint(0, num_entities, (batch_size,))
        for i in range(3, num_components):
            facts[:, i] = torch.randint(0, max(num_entities, num_relations), (batch_size,))
        return facts

    # ========================================================================
    # Tests for forward method
    # ========================================================================

    def test_forward_output_shape(self, kge: KGE, sample_facts: torch.Tensor) -> None:
        """Test that forward returns correct output shape."""
        scores = kge.forward(sample_facts)
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (sample_facts.shape[0],)
        assert scores.dtype in (torch.float32, torch.float64)

    def test_forward_single_fact(self, kge: KGE) -> None:
        """Test forward with single fact."""
        single_fact = torch.tensor([[0, 1, 2]], dtype=torch.long)
        scores = kge.forward(single_fact)
        assert scores.shape == (1,)

    def test_forward_large_batch(self, kge: KGE, large_facts: torch.Tensor) -> None:
        """Test forward with larger batch size."""
        scores = kge.forward(large_facts)
        assert scores.shape == (large_facts.shape[0],)

    def test_forward_extra_dimensions(self, kge: KGE, facts_with_extra_dims: torch.Tensor) -> None:
        """Test forward with facts having M > 3 dimensions."""
        scores = kge.forward(facts_with_extra_dims)
        assert scores.shape == (facts_with_extra_dims.shape[0],)

    def test_forward_preserves_device(self, kge: KGE, sample_facts: torch.Tensor) -> None:
        """Test that forward preserves input device."""
        if torch.cuda.is_available():
            device_facts = sample_facts.cuda()
            kge = kge.cuda()
            scores = kge.forward(device_facts)
            assert scores.device == device_facts.device

    def test_forward_asserts_2d_tensor(self, kge: KGE) -> None:
        """Test that forward asserts 2D tensor input."""
        with pytest.raises(AssertionError, match="Expected 2D tensor"):
            kge.forward(torch.tensor([0, 1, 2], dtype=torch.long))

    def test_forward_asserts_min_columns(self, kge: KGE) -> None:
        """Test that forward asserts at least 3 columns."""
        with pytest.raises(AssertionError, match="Expected at least 3 columns"):
            kge.forward(torch.tensor([[0, 1]], dtype=torch.long))

    # ========================================================================
    # Tests for score method
    # ========================================================================

    def test_score_output_shape(self, kge: KGE, sample_facts: torch.Tensor) -> None:
        """Test that score returns correct output shape."""
        scores = kge.score(sample_facts)
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (sample_facts.shape[0],)

    def test_score_equals_forward(self, kge: KGE, sample_facts: torch.Tensor) -> None:
        """Test that score method returns same results as forward."""
        forward_scores = kge.forward(sample_facts)
        score_scores = kge.score(sample_facts)
        torch.testing.assert_close(forward_scores, score_scores)

    # ========================================================================
    # Tests for score_relations method
    # ========================================================================

    def test_score_relations_output_shape(self, kge: KGE, kge_config: dict[str, int]) -> None:
        """Test that score_relations returns correct output shape."""
        batch_size = 5
        head = torch.randint(0, kge_config["num_entities"], (batch_size,))
        tail = torch.randint(0, kge_config["num_entities"], (batch_size,))
        scores = kge.score_relations(head, tail)
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (batch_size, kge_config["num_relations"])

    def test_score_relations_single_pair(self, kge: KGE, kge_config: dict[str, int]) -> None:
        """Test score_relations with single (head, tail) pair."""
        head = torch.tensor([0])
        tail = torch.tensor([1])
        scores = kge.score_relations(head, tail)
        assert scores.shape == (1, kge_config["num_relations"])

    def test_score_relations_large_batch(self, kge: KGE, kge_config: dict[str, int]) -> None:
        """Test score_relations with larger batch size."""
        batch_size = 20
        head = torch.randint(0, kge_config["num_entities"], (batch_size,))
        tail = torch.randint(0, kge_config["num_entities"], (batch_size,))
        scores = kge.score_relations(head, tail)
        assert scores.shape == (batch_size, kge_config["num_relations"])

    def test_score_relations_preserves_device(self, kge: KGE) -> None:
        """Test that score_relations preserves input device."""
        if torch.cuda.is_available():
            head = torch.tensor([0, 1]).cuda()
            tail = torch.tensor([2, 3]).cuda()
            kge = kge.cuda()
            scores = kge.score_relations(head, tail)
            assert scores.device == head.device

    # ========================================================================
    # Tests for score_subjects method
    # ========================================================================

    def test_score_subjects_output_shape(self, kge: KGE, kge_config: dict[str, int]) -> None:
        """Test that score_subjects returns correct output shape."""
        batch_size = 5
        relation = torch.randint(0, kge_config["num_relations"], (batch_size,))
        tail = torch.randint(0, kge_config["num_entities"], (batch_size,))
        scores = kge.score_subjects(relation, tail)
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (batch_size, kge_config["num_entities"])

    def test_score_subjects_single_pair(self, kge: KGE, kge_config: dict[str, int]) -> None:
        """Test score_subjects with single (relation, tail) pair."""
        relation = torch.tensor([0])
        tail = torch.tensor([1])
        scores = kge.score_subjects(relation, tail)
        assert scores.shape == (1, kge_config["num_entities"])

    def test_score_subjects_large_batch(self, kge: KGE, kge_config: dict[str, int]) -> None:
        """Test score_subjects with larger batch size."""
        batch_size = 20
        relation = torch.randint(0, kge_config["num_relations"], (batch_size,))
        tail = torch.randint(0, kge_config["num_entities"], (batch_size,))
        scores = kge.score_subjects(relation, tail)
        assert scores.shape == (batch_size, kge_config["num_entities"])

    def test_score_subjects_preserves_device(
        self,
        kge: KGE,
    ) -> None:
        """Test that score_subjects preserves input device."""
        if torch.cuda.is_available():
            relation = torch.tensor([0, 1]).cuda()
            tail = torch.tensor([2, 3]).cuda()
            kge = kge.cuda()
            scores = kge.score_subjects(relation, tail)
            assert scores.device == relation.device

    # ========================================================================
    # Tests for score_objects method
    # ========================================================================

    def test_score_objects_output_shape(self, kge: KGE, kge_config: dict[str, int]) -> None:
        """Test that score_objects returns correct output shape."""
        batch_size = 5
        head = torch.randint(0, kge_config["num_entities"], (batch_size,))
        relation = torch.randint(0, kge_config["num_relations"], (batch_size,))
        scores = kge.score_objects(head, relation)
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (batch_size, kge_config["num_entities"])

    def test_score_objects_single_pair(self, kge: KGE, kge_config: dict[str, int]) -> None:
        """Test score_objects with single (head, relation) pair."""
        head = torch.tensor([0])
        relation = torch.tensor([1])
        scores = kge.score_objects(head, relation)
        assert scores.shape == (1, kge_config["num_entities"])

    def test_score_objects_large_batch(self, kge: KGE, kge_config: dict[str, int]) -> None:
        """Test score_objects with larger batch size."""
        batch_size = 20
        head = torch.randint(0, kge_config["num_entities"], (batch_size,))
        relation = torch.randint(0, kge_config["num_relations"], (batch_size,))
        scores = kge.score_objects(head, relation)
        assert scores.shape == (batch_size, kge_config["num_entities"])

    def test_score_objects_preserves_device(self, kge: KGE) -> None:
        """Test that score_objects preserves input device."""
        if torch.cuda.is_available():
            head = torch.tensor([0, 1]).cuda()
            relation = torch.tensor([2, 3]).cuda()
            kge = kge.cuda()
            scores = kge.score_objects(head, relation)
            assert scores.device == head.device

    # ========================================================================
    # Tests for Embeddings
    # ========================================================================

    def test_entity_embeddings_shape(self, kge: KGE, kge_config: dict[str, int]) -> None:
        """Test that entity embeddings have correct shape."""
        assert kge.entity_embeddings.weight.shape == (
            kge_config["num_entities"],
            kge_config["embedding_dim"],
        )

    def test_relation_embeddings_shape(self, kge: KGE, kge_config: dict[str, int]) -> None:
        """Test that relation embeddings have correct shape."""
        assert kge.relation_embeddings.weight.shape == (
            kge_config["num_relations"],
            kge_config["embedding_dim"],
        )

    def test_embeddings_are_trainable(self, kge: KGE) -> None:
        """Test that embeddings are trainable parameters."""
        assert kge.entity_embeddings.weight.requires_grad
        assert kge.relation_embeddings.weight.requires_grad

    # ========================================================================
    # Edge Cases
    # ========================================================================

    def test_forward_empty_batch(self, kge: KGE) -> None:
        """Test forward with empty batch."""
        empty_facts = torch.empty(0, 3, dtype=torch.long)
        scores = kge.forward(empty_facts)
        assert scores.shape == (0,)

    def test_forward_boundary_indices(self, kge: KGE, kge_config: dict[str, int]) -> None:
        """Test forward with boundary entity/relation indices."""
        max_entity = kge_config["num_entities"] - 1
        max_relation = kge_config["num_relations"] - 1
        boundary_facts = torch.tensor(
            [[0, 0, 0], [max_entity, max_relation, max_entity]],
            dtype=torch.long,
        )
        scores = kge.forward(boundary_facts)
        assert scores.shape == (2,)

    def test_score_relations_empty_batch(self, kge: KGE, kge_config: dict[str, int]) -> None:
        """Test score_relations with empty batch."""
        empty_head = torch.empty(0, dtype=torch.long)
        empty_tail = torch.empty(0, dtype=torch.long)
        scores = kge.score_relations(empty_head, empty_tail)
        assert scores.shape == (0, kge_config["num_relations"])

    def test_score_subjects_empty_batch(self, kge: KGE, kge_config: dict[str, int]) -> None:
        """Test score_subjects with empty batch."""
        empty_relation = torch.empty(0, dtype=torch.long)
        empty_tail = torch.empty(0, dtype=torch.long)
        scores = kge.score_subjects(empty_relation, empty_tail)
        assert scores.shape == (0, kge_config["num_entities"])

    def test_score_objects_empty_batch(self, kge: KGE, kge_config: dict[str, int]) -> None:
        """Test score_objects with empty batch."""
        empty_head = torch.empty(0, dtype=torch.long)
        empty_relation = torch.empty(0, dtype=torch.long)
        scores = kge.score_objects(empty_head, empty_relation)
        assert scores.shape == (0, kge_config["num_entities"])
