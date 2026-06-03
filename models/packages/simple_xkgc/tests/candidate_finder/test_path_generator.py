# ruff: noqa: PLR2004
from collections.abc import Callable

import numpy as np
import pytest
import torch
from hakken_ml_toolkit.ml_base_structures.fact import FactIndexList, assert_is_fact_index_list
from kge.models.config import KGEConfig
from kge.models.random import RandomKGE

from hakken_explainer.candidate_finder.path_generator import (
    KGEPathGenerator,
    PathGenerator,
)

# ============================================================================
# IMPLEMENTATIONS LIST - Add all implementations you want to test here
# ============================================================================

NUM_RELATIONS = 12
NUM_ENTITIES = 1000
kge = RandomKGE(
    KGEConfig(num_entities=NUM_ENTITIES, num_relations=NUM_RELATIONS, embedding_dim=128)
)

PATH_GENERATOR_FACTORIES: list[Callable[[], PathGenerator]] = [
    # lambda: RandomPathGenerator(
    #     entity_indices=list(range(NUM_ENTITIES)), relation_indices=list(range(NUM_RELATIONS))
    # ),
    lambda: KGEPathGenerator(
        entity_indices=list(range(NUM_ENTITIES)),
        relation_indices=list(range(NUM_RELATIONS)),
        model=kge,
    ),
]


# ============================================================================
# HELPER FUNCTION
# ============================================================================


def create_generator_with_device(
    factory: Callable[[], PathGenerator], device: str | torch.device = "cpu"
) -> PathGenerator:
    """Helper to create a generator with a specific device.

    This handles the case where we want to test device changes,
    but the factory might already set a device.
    """
    generator = factory()
    generator.to_device(device)
    return generator


# ============================================================================
# TESTS FOR __init__
# ============================================================================


@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_init_default_device(path_generator_factory: Callable[[], PathGenerator]) -> None:
    """Test PathGenerator initialization with default device."""
    generator = path_generator_factory()
    # Device should be set (either default or from factory)
    assert hasattr(generator, "device")
    assert generator.device in ["cpu", torch.device("cpu")]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_init_custom_device(path_generator_factory: Callable[[], PathGenerator]) -> None:
    """Test PathGenerator initialization with custom device."""
    # Create with factory, then change device
    generator = create_generator_with_device(path_generator_factory, "cuda")
    assert generator.device == "cuda" or generator.device == torch.device("cuda")


@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_init_torch_device(path_generator_factory: Callable[[], PathGenerator]) -> None:
    """Test PathGenerator initialization with torch.device."""
    device = torch.device("cpu")
    generator = create_generator_with_device(path_generator_factory, device)
    assert generator.device in (device, "cpu")


# ============================================================================
# TESTS FOR to_device
# ============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_to_device_string(path_generator_factory: Callable[[], PathGenerator]) -> None:
    """Test changing device using string."""
    generator = path_generator_factory()
    generator.to_device("cuda")
    assert generator.device == "cuda" or generator.device == torch.device("cuda")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_to_device_torch_device(path_generator_factory: Callable[[], PathGenerator]) -> None:
    """Test changing device using torch.device."""
    generator = path_generator_factory()
    device = torch.device("cuda")
    generator.to_device(device)
    assert generator.device == device


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_to_device_cpu(path_generator_factory: Callable[[], PathGenerator]) -> None:
    """Test changing device back to CPU."""
    generator = path_generator_factory()
    generator.to_device("cuda")
    generator.to_device("cpu")
    assert generator.device == "cpu" or generator.device == torch.device("cpu")


# ============================================================================
# TESTS FOR convert_to_numpy
# ============================================================================


@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_convert_to_numpy_list(path_generator_factory: Callable[[], PathGenerator]) -> None:
    """Test convert_to_numpy with list input."""
    generator = path_generator_factory()
    input_list: list[int] = [1, 2, 3, 4, 5]
    result = generator.convert_to_numpy(input_list)

    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1, 2, 3, 4, 5]))
    assert result.dtype in (np.int64, np.int32)


@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_convert_to_numpy_torch_tensor(path_generator_factory: Callable[[], PathGenerator]) -> None:
    """Test convert_to_numpy with torch.Tensor input."""
    generator = path_generator_factory()
    tensor = torch.tensor([1, 2, 3, 4, 5])
    result = generator.convert_to_numpy(tensor)

    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1, 2, 3, 4, 5]))


@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_convert_to_numpy_torch_tensor_cuda(
    path_generator_factory: Callable[[], PathGenerator],
) -> None:
    """Test convert_to_numpy with CUDA torch.Tensor input."""
    generator = path_generator_factory()
    if torch.cuda.is_available():
        tensor = torch.tensor([1, 2, 3], device="cuda")
        result = generator.convert_to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1, 2, 3]))


@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_convert_to_numpy_numpy_array(path_generator_factory: Callable[[], PathGenerator]) -> None:
    """Test convert_to_numpy with numpy array input."""
    generator = path_generator_factory()
    input_array = np.array([1, 2, 3, 4, 5])
    result = generator.convert_to_numpy(input_array)

    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, input_array)
    # Should return the same array (or a view)
    assert result is input_array or np.array_equal(result, input_array)


@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_convert_to_numpy_empty_list(path_generator_factory: Callable[[], PathGenerator]) -> None:
    """Test convert_to_numpy with empty list."""
    generator = path_generator_factory()
    result = generator.convert_to_numpy([])

    assert isinstance(result, np.ndarray)
    assert result.shape == (0,)


@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_convert_to_numpy_unsupported_type(
    path_generator_factory: Callable[[], PathGenerator],
) -> None:
    """Test convert_to_numpy with unsupported type raises TypeError."""
    generator = path_generator_factory()
    with pytest.raises(TypeError, match="Unsupported type"):
        generator.convert_to_numpy("invalid")  # type: ignore


# ============================================================================
# TESTS FOR generate_facts
# ============================================================================


@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_generate_facts_with_source_list(
    path_generator_factory: Callable[[], PathGenerator],
) -> None:
    """Test generate_facts with source as list."""
    generator = path_generator_factory()
    source: list[int] = [1, 2, 3]
    facts = generator.generate_facts(source=source)

    assert_is_fact_index_list(facts)


@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_generate_facts_with_source_tensor(
    path_generator_factory: Callable[[], PathGenerator],
) -> None:
    """Test generate_facts with source as torch.Tensor."""
    generator = path_generator_factory()
    source = torch.tensor([1, 2, 3])
    facts = generator.generate_facts(source=source)

    assert_is_fact_index_list(facts)


@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_generate_facts_with_target(path_generator_factory: Callable[[], PathGenerator]) -> None:
    """Test generate_facts with target parameter."""
    generator = path_generator_factory()
    target: list[int] = [4, 5, 6]
    facts = generator.generate_facts(target=target)
    assert_is_fact_index_list(facts)


# ============================================================================
# TESTS FOR generate_paths (abstract method)
# ============================================================================


@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_generate_paths_default(path_generator_factory: Callable[[], PathGenerator]) -> None:
    """Test generate_paths with default parameters."""
    generator = path_generator_factory()
    paths = generator.generate_paths(source=1, target=2, num_hops=3)

    assert isinstance(paths, list)
    assert len(paths) == 1
    for path in paths:
        assert_is_fact_index_list(path)


@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_generate_paths_num_paths(path_generator_factory: Callable[[], PathGenerator]) -> None:
    """Test generate_paths with custom num_paths."""
    generator = path_generator_factory()
    paths = generator.generate_paths(source=1, target=2, num_hops=3, num_paths=5)

    assert isinstance(paths, list)
    assert len(paths) == 5
    for path in paths:
        assert_is_fact_index_list(path, length=3)


@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_generate_paths_previous_generated_paths(
    path_generator_factory: Callable[[], PathGenerator],
) -> None:
    """Test generate_paths with previous_generated_paths parameter."""
    generator = path_generator_factory()
    previous_paths: list[FactIndexList] = [[]]
    paths = generator.generate_paths(
        source=1, target=2, num_hops=3, previous_generated_paths=previous_paths
    )

    assert isinstance(paths, list)

    for path in paths:
        assert_is_fact_index_list(path, length=3)


@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_generate_paths_allowed_relations(
    path_generator_factory: Callable[[], PathGenerator],
) -> None:
    """Test generate_paths with allowed_relations parameter."""
    generator = path_generator_factory()
    allowed_relations: list[int] = [1, 3]
    paths = generator.generate_paths(
        source=1, target=2, num_hops=3, allowed_relations=allowed_relations
    )

    assert isinstance(paths, list)

    for path in paths:
        assert_is_fact_index_list(path, length=3)
        for fact in path:
            assert fact[1] in allowed_relations


@pytest.mark.parametrize("num_hops", [1, 2, 3, 5], ids=lambda n: f"hops={n}")
@pytest.mark.parametrize("path_generator_factory", PATH_GENERATOR_FACTORIES)
def test_generate_paths_various_hops(
    path_generator_factory: Callable[[], PathGenerator],
    num_hops: int,
) -> None:
    """Test generate_paths with various num_hops values."""
    generator = path_generator_factory()
    paths = generator.generate_paths(source=1, target=2, num_hops=num_hops)
    assert isinstance(paths, list)

    for path in paths:
        assert_is_fact_index_list(path, length=num_hops)


@pytest.mark.parametrize(
    "path_generator_factory,source,target,num_hops",
    [(factory, 0, 1, 1) for factory in PATH_GENERATOR_FACTORIES]
    + [(factory, 1, 10, 2) for factory in PATH_GENERATOR_FACTORIES]
    + [(factory, 5, 20, 3) for factory in PATH_GENERATOR_FACTORIES],
)
def test_generate_paths_parametrized(
    path_generator_factory: Callable[[], PathGenerator], source: int, target: int, num_hops: int
) -> None:
    """Parameterized test for generate_paths with various source/target/hops."""
    generator = path_generator_factory()
    paths = generator.generate_paths(source=source, target=target, num_hops=num_hops)

    assert isinstance(paths, list)

    for path in paths:
        assert_is_fact_index_list(path, length=num_hops)
