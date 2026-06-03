import pytest
import torch

from hakken_models.core.constants import FactComponent
from hakken_models.negative_samplers import NegativeSampler

# ============================================================================
# Parameter Sets for NegativeSampler Constructor
# ============================================================================

NEGATIVE_SAMPLER_CONFIGS = [
    {"num_entities": 100, "num_relations": 10, "corruption_scheme": None, "fact_validator": None},
    {
        "num_entities": 200,
        "num_relations": 20,
        "corruption_scheme": [FactComponent.SUBJECT],
        "fact_validator": None,
    },
    {
        "num_entities": 150,
        "num_relations": 15,
        "corruption_scheme": [FactComponent.OBJECT],
        "fact_validator": None,
    },
]

# ============================================================================
# Base Test class
# ============================================================================


class BaseNegativeSamplerTests:
    """
    Base test class focusing on functional behavior of NegativeSampler.
    Tests both _corrupt_facts_once and corrupt_facts methods.
    """

    __test__ = False  # Prevent pytest from collecting this as a test

    @pytest.fixture(params=NEGATIVE_SAMPLER_CONFIGS)
    def sampler_config(self, request):
        """Parametrized configuration fixture."""
        return request.param

    @pytest.fixture
    def sampler(self, sampler_config: dict) -> NegativeSampler:
        """Default sampler instance without validator."""
        raise NotImplementedError("Subclass must provide sampler")

    # ========================================================================
    # Test Data Fixtures (depend on sampler_config)
    # ========================================================================

    @pytest.fixture
    def sample_facts(self, sampler_config) -> torch.Tensor:
        """Sample facts tensor for testing, respecting entity/relation bounds."""
        num_entities = sampler_config["num_entities"]
        num_relations = sampler_config.get("num_relations", num_entities)

        # Generate facts within valid bounds
        return torch.tensor(
            [
                [0, 0, 1],
                [min(1, num_entities - 1), min(1, num_relations - 1), min(2, num_entities - 1)],
                [min(2, num_entities - 1), min(2, num_relations - 1), min(3, num_entities - 1)],
            ],
            dtype=torch.long,
        )

    @pytest.fixture
    def large_facts(self, sampler_config) -> torch.Tensor:
        """Larger facts tensor for testing, respecting entity/relation bounds."""
        num_entities = sampler_config["num_entities"]
        num_relations = sampler_config.get("num_relations", num_entities)

        # Generate random facts within valid bounds
        batch_size = 20
        facts = torch.zeros(batch_size, 3, dtype=torch.long)
        facts[:, 0] = torch.randint(0, num_entities, (batch_size,))
        facts[:, 1] = torch.randint(0, num_relations, (batch_size,))
        facts[:, 2] = torch.randint(0, num_entities, (batch_size,))
        return facts

    @pytest.fixture
    def facts_with_extra_dims(self, sampler_config) -> torch.Tensor:
        """Facts with extra dimensions (M > 3), respecting entity/relation bounds."""
        num_entities = sampler_config["num_entities"]
        num_relations = sampler_config.get("num_relations", num_entities)

        batch_size = 10
        num_components = 5
        facts = torch.zeros(batch_size, num_components, dtype=torch.long)

        # First column: subject (entity)
        facts[:, 0] = torch.randint(0, num_entities, (batch_size,))
        # Second column: relation (if applicable)
        if num_components > 1:
            facts[:, 1] = torch.randint(0, num_relations, (batch_size,))
        # Last column: object (entity)
        facts[:, -1] = torch.randint(0, num_entities, (batch_size,))
        # Middle columns: can be additional components
        for i in range(2, num_components - 1):
            facts[:, i] = torch.randint(0, max(num_entities, num_relations), (batch_size,))

        return facts

    # ========================================================================
    # Tests for _corrupt_facts_once (Core Corruption Logic)
    # ========================================================================

    def test_corrupt_facts_once_output_shape(
        self, sampler: NegativeSampler, sample_facts: torch.Tensor
    ) -> None:
        """Test that _corrupt_facts_once returns correct output shape."""
        num_negatives = 3
        corrupted = sampler._corrupt_facts_once(sample_facts, num_negatives)

        assert isinstance(corrupted, torch.Tensor)
        assert corrupted.shape == (sample_facts.shape[0], num_negatives, sample_facts.shape[1])

    def test_corrupt_facts_once_single_negative(
        self, sampler: NegativeSampler, sample_facts: torch.Tensor
    ) -> None:
        """Test _corrupt_facts_once with single negative sample."""
        corrupted = sampler._corrupt_facts_once(sample_facts, num_negatives=1)
        assert corrupted.shape == (sample_facts.shape[0], 1, sample_facts.shape[1])

    def test_corrupt_facts_once_multiple_negatives(
        self, sampler: NegativeSampler, sample_facts: torch.Tensor
    ) -> None:
        """Test _corrupt_facts_once with multiple negative samples."""
        num_negatives = 5
        corrupted = sampler._corrupt_facts_once(sample_facts, num_negatives=num_negatives)
        assert corrupted.shape == (sample_facts.shape[0], num_negatives, sample_facts.shape[1])

    def test_corrupt_facts_once_large_batch(
        self, sampler: NegativeSampler, large_facts: torch.Tensor
    ) -> None:
        """Test _corrupt_facts_once with larger batch size."""
        corrupted = sampler._corrupt_facts_once(large_facts, num_negatives=2)
        assert corrupted.shape == (large_facts.shape[0], 2, large_facts.shape[1])

    def test_corrupt_facts_once_extra_dimensions(
        self, sampler: NegativeSampler, facts_with_extra_dims: torch.Tensor
    ) -> None:
        """Test _corrupt_facts_once with facts having M > 3 dimensions."""
        corrupted = sampler._corrupt_facts_once(facts_with_extra_dims, num_negatives=2)
        assert corrupted.shape == (
            facts_with_extra_dims.shape[0],
            2,
            facts_with_extra_dims.shape[1],
        )

    def test_corrupt_facts_once_preserves_dtype(
        self, sampler: NegativeSampler, sample_facts: torch.Tensor
    ) -> None:
        """Test that _corrupt_facts_once preserves input dtype."""
        corrupted = sampler._corrupt_facts_once(sample_facts, num_negatives=2)
        assert corrupted.dtype == sample_facts.dtype

    def test_corrupt_facts_once_preserves_device(
        self, sampler: NegativeSampler, sample_facts: torch.Tensor
    ) -> None:
        """Test that _corrupt_facts_once preserves input device."""
        if torch.cuda.is_available():
            device_facts = sample_facts.cuda()
            corrupted = sampler._corrupt_facts_once(device_facts, num_negatives=2)
            assert corrupted.device == device_facts.device

    def test_corrupt_facts_once_entity_bounds(
        self, sampler: NegativeSampler, sample_facts: torch.Tensor
    ) -> None:
        """Test that _corrupt_facts_once produces entities within valid bounds."""
        corrupted = sampler._corrupt_facts_once(sample_facts, num_negatives=5)

        # Check all entity columns (subject and object)
        subject_values = corrupted[:, :, 0].flatten()
        object_values = corrupted[:, :, -1].flatten()

        assert torch.all(subject_values >= 0)
        assert torch.all(subject_values < sampler.num_entities)
        assert torch.all(object_values >= 0)
        assert torch.all(object_values < sampler.num_entities)

    def test_corrupt_facts_once_relation_bounds(
        self, sampler: NegativeSampler, sample_facts: torch.Tensor
    ) -> None:
        """Test that _corrupt_facts_once produces relations within valid bounds."""
        if sampler.num_relations is not None:
            corrupted = sampler._corrupt_facts_once(sample_facts, num_negatives=5)
            # Assuming middle column is relation (adjust if schema differs)
            relation_values = corrupted[:, :, 1].flatten()
            assert torch.all(relation_values >= 0)
            assert torch.all(relation_values < sampler.num_relations)

    def test_corrupt_facts_once_corruption_occurs(
        self, sampler: NegativeSampler, sample_facts: torch.Tensor
    ) -> None:
        """Test that _corrupt_facts_once actually corrupts facts (changes values)."""
        corrupted = sampler._corrupt_facts_once(sample_facts, num_negatives=3)

        # At least some values should differ from original
        # Check each fact individually
        for fact_idx in range(sample_facts.shape[0]):
            original_fact = sample_facts[fact_idx]
            corrupted_for_fact = corrupted[fact_idx]

            # At least one negative sample should differ from original
            differs = False
            for neg_idx in range(corrupted_for_fact.shape[0]):
                if not torch.equal(original_fact, corrupted_for_fact[neg_idx]):
                    differs = True
                    break
            assert differs, f"Fact {fact_idx} was not corrupted"

    # ========================================================================
    # Tests for corrupt_facts (Public API with Validation)
    # ========================================================================

    def test_corrupt_facts_without_validator(
        self, sampler: NegativeSampler, sample_facts: torch.Tensor
    ) -> None:
        """Test corrupt_facts without validator (should bypass validation)."""
        num_negatives = 2
        corrupted = sampler.corrupt_facts(sample_facts, num_negatives=num_negatives)

        assert corrupted.shape == (sample_facts.shape[0], num_negatives, sample_facts.shape[1])

    def test_corrupt_facts_without_validator_single_attempt(
        self, sampler: NegativeSampler, sample_facts: torch.Tensor
    ) -> None:
        """Test corrupt_facts with num_attempts=1 (should bypass validation)."""
        num_negatives = 2
        corrupted = sampler.corrupt_facts(sample_facts, num_negatives=num_negatives, num_attempts=1)

        assert corrupted.shape == (sample_facts.shape[0], num_negatives, sample_facts.shape[1])

    # ========================================================================
    # Edge Cases
    # ========================================================================

    def test_corrupt_facts_single_fact(self, sampler: NegativeSampler) -> None:
        """Test corrupt_facts with single fact."""
        single_fact = torch.tensor([[0, 1, 2]], dtype=torch.long)
        corrupted = sampler.corrupt_facts(single_fact, num_negatives=3)
        assert corrupted.shape == (1, 3, 3)

    def test_corrupt_facts_empty_batch(self, sampler: NegativeSampler) -> None:
        """Test corrupt_facts with empty batch."""
        empty_facts = torch.empty(0, 3, dtype=torch.long)
        corrupted = sampler.corrupt_facts(empty_facts, num_negatives=2)
        assert corrupted.shape == (0, 2, 3)

    def test_corrupt_facts_zero_negatives(
        self, sampler: NegativeSampler, sample_facts: torch.Tensor
    ) -> None:
        """Test corrupt_facts with zero negatives (edge case)."""
        corrupted = sampler.corrupt_facts(sample_facts, num_negatives=0)
        assert corrupted.shape == (sample_facts.shape[0], 0, sample_facts.shape[1])
