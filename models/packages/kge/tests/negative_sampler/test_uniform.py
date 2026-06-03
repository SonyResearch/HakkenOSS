import pytest
import torch
from hakken_ml_toolkit.ml_base_structures.data_generator import DummyDataGenerator
from pydantic import ValidationError

from kge.common.constants import TargetType
from kge.common.exceptions import SplitsError
from kge.common.validation import is_long_tensor_with_dim
from kge.negative_sampler import (
    UniformNegativeSampler,
    UniformNegativeSamplerConfig,
)


@pytest.mark.parametrize("num_entities", [100, 2000])
@pytest.mark.parametrize("num_relations", [2, 10])
@pytest.mark.parametrize("num_negatives", [1])
@pytest.mark.parametrize("corruption_scheme", [[TargetType.OBJECT], [TargetType.SUBJECT]])
@pytest.mark.parametrize("filter_triples", [["train"], ["train", "valid"]])
@pytest.mark.parametrize("device", ["cpu"])
def test_uniform_negative_sampler(
    num_entities: int,
    num_relations: int,
    num_negatives: int,
    corruption_scheme: list[TargetType],
    filter_triples: list[str],
    device: str,
):
    config = UniformNegativeSamplerConfig(
        num_negatives=num_negatives,
        filter_triples=filter_triples,
        corruption_scheme=corruption_scheme,
    )
    kg = DummyDataGenerator.knowledge_graph(
        batch_size=1000,
        num_entities=num_entities,
        num_relations=num_relations,
        device="cpu",
    )

    sampler = UniformNegativeSampler(config)
    sampler.set_up(kg, device)

    assert isinstance(sampler, UniformNegativeSampler)
    assert sampler.config == config
    assert sampler.kg == kg
    assert sampler.sampler is not None

    # Test corruption scheme
    expected_scheme = ["tail"] if corruption_scheme == ["object"] else ["head"]
    assert sampler.sampler.corruption_scheme == expected_scheme

    # Test corrupt_batch
    positive_batch = torch.randint(0, min(num_entities, num_relations), (10, 3))
    corrupted_batch = sampler.corrupt_batch(positive_batch)
    assert is_long_tensor_with_dim(corrupted_batch, dim=3)
    assert corrupted_batch.shape == (10, num_negatives, 3)

    # Test filtered triples
    if filter_triples:
        filtered_triples = sampler.get_filtered_triples()

        assert is_long_tensor_with_dim(filtered_triples, dim=2)
        expected_shape = sum(
            kg.facts_dict[split].data.shape[0] for split in filter_triples if split in kg.facts_dict
        )
        assert filtered_triples.shape[0] == expected_shape
    else:
        with pytest.raises(SplitsError):
            sampler.get_filtered_triples()


def test_uniform_negative_sampler_invalid_corruption_scheme():
    with pytest.raises(ValidationError):
        _ = UniformNegativeSamplerConfig(
            num_negatives=5,
            filter_triples=["train"],
            corruption_scheme=["invalid"],  # type: ignore
        )


@pytest.mark.parametrize("num_entities", [100, 2000])
@pytest.mark.parametrize("num_relations", [2, 10])
@pytest.mark.parametrize("device", ["cpu"])
def test_uniform_negative_sampler_entity_relation_counts(
    num_entities: int, num_relations: int, device: str
):
    config = UniformNegativeSamplerConfig(
        num_negatives=5, filter_triples=["train"], corruption_scheme=[TargetType.OBJECT]
    )
    kg = DummyDataGenerator.knowledge_graph(
        batch_size=1000,
        num_entities=num_entities,
        num_relations=num_relations,
        device="cpu",
    )
    sampler = UniformNegativeSampler(config)
    sampler.set_up(kg, device)

    assert sampler.sampler.num_entities == num_entities
    assert sampler.sampler.num_relations == num_relations

    # Test that corrupted entities are within the correct range
    positive_batch = torch.randint(0, min(num_entities, num_relations), (10, 3))
    corrupted_batch = sampler.corrupt_batch(positive_batch)
    assert torch.all(corrupted_batch[:, :, 2] < num_entities)  # Assuming object corruption
