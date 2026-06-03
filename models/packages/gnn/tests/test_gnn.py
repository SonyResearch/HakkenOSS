from typing import Any

import pytest
import torch
import torch_geometric.data as pygd
from hakken_ml_toolkit.ml_utils.constants import ActivationType

from gnn import (
    GAT,
    GCN,
    GIN,
    RGCN,
    GATConfig,
    GCNConfig,
    GINConfig,
    GraphSAGE,
    GraphSAGEConfig,
    RGCNConfig,
)
from gnn.architectures.base import GNNI, GNNConfig
from gnn.common.constants import PoolingType, StageType

# Type aliases
ModelType = type[GNNI]
ConfigType = type[GNNConfig]
ModelConfigTuple = tuple[ModelType, GNNConfig]


num_relations = 10


# Model configurations
@pytest.fixture(
    params=[
        (GCN, GCNConfig, {}),
        (GAT, GATConfig, {"heads": 4, "edge_dim": 0}),
        (GIN, GINConfig, {"eps": 0.5, "train_eps": False}),
        (
            RGCN,
            RGCNConfig,
            {"num_relations": num_relations, "num_bases": None, "aggr": "mean"},
        ),
        (GraphSAGE, GraphSAGEConfig, {"num_sampled_edges_per_hop": 10}),
    ]
)
def model_config(
    request: pytest.FixtureRequest,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    device: str,
) -> ModelConfigTuple:
    model_class, config_class, extra_params = request.param
    base_params: dict[str, Any] = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": hidden_dim,
        "activation": ActivationType.RELU,
        "stage_type": StageType.SKIPSUM,
        "device": device,
    }
    # Merge base parameters with model-specific parameters
    config_params = {**base_params, **extra_params}
    return model_class, config_class(**config_params)


# Existing fixtures with type annotations
@pytest.fixture
def device() -> str:
    return "cpu"


@pytest.fixture
def batch_size() -> int:
    return 3


@pytest.fixture
def num_nodes() -> int:
    return 5


@pytest.fixture
def input_dim() -> int:
    return 10


@pytest.fixture
def hidden_dim() -> int:
    return 16


@pytest.fixture
def output_dim() -> int:
    return 4


@pytest.fixture
def edge_index(num_nodes: int) -> torch.Tensor:
    sources = torch.tensor(list(range(num_nodes - 1)), dtype=torch.long)
    targets = torch.tensor(list(range(1, num_nodes)), dtype=torch.long)
    return torch.stack([torch.cat([sources, targets]), torch.cat([targets, sources])])


@pytest.fixture
def edge_attr(edge_index: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    num_edges = edge_index.size(1)
    return torch.randn(num_edges, 1, device=device)


@pytest.fixture
def edge_type(edge_index: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    num_edges = edge_index.size(1)
    return torch.randint(low=0, high=num_relations, size=(num_edges,), device=device)


@pytest.fixture
def sample_batch(
    batch_size: int,
    num_nodes: int,
    input_dim: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_type: torch.Tensor,
    device: str,
) -> pygd.Batch:
    data_list = []
    for _i in range(batch_size):
        x = torch.randn(num_nodes, input_dim, device=device)
        data = pygd.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)
        data_list.append(data)
    return pygd.Batch.from_data_list(data_list)


# Generic test functions with type annotations
def test_gnn_init(model_config: ModelConfigTuple) -> None:
    model_class, config = model_config
    model = model_class(config)

    assert hasattr(model, "pre_nn")
    assert hasattr(model, "gnn")
    assert hasattr(model, "post_nn")
    assert isinstance(model, GNNI)


def test_forward_pass(
    model_config: ModelConfigTuple, sample_batch: pygd.Batch, output_dim: int
) -> None:
    model_class, config = model_config
    model = model_class(config)
    output = model(sample_batch)

    expected_nodes = sample_batch.x.size(0)
    assert output.shape == (expected_nodes, output_dim)
    assert output.device.type == config.device


@pytest.mark.parametrize("use_pooling", [True, False])
def test_pooling(
    model_config: ModelConfigTuple,
    sample_batch: pygd.Batch,
    output_dim: int,
    use_pooling: bool,
) -> None:
    model_class, config = model_config
    if use_pooling:
        config.pooling = [PoolingType.MEAN]

    model = model_class(config)
    output = model(sample_batch)

    expected_size = sample_batch.num_graphs if use_pooling else sample_batch.num_nodes
    assert output.shape == (expected_size, output_dim)


@pytest.mark.parametrize("num_layers", [0, 1, 2, 3])
def test_gnn_layers(
    model_config: ModelConfigTuple, sample_batch: pygd.Batch, num_layers: int
) -> None:
    model_class, config = model_config
    config.num_layers_gnn = num_layers
    model = model_class(config)

    if num_layers == 0:
        assert len(model.gnn) == 1
        assert isinstance(model.gnn[0], torch.nn.Identity)
    else:
        assert len(model.gnn) == num_layers

    output = model(sample_batch)
    assert output is not None


def test_skipsum_connections(model_config: ModelConfigTuple, sample_batch: pygd.Batch) -> None:
    model_class, config = model_config
    config.stage_type = StageType.SKIPSUM
    config.num_layers_gnn = 2
    model = model_class(config)

    assert model.lin_skipsum is not None
    assert len(model.lin_skipsum) == 2

    output = model(sample_batch)
    assert output is not None


@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_regularization(
    model_config: ModelConfigTuple,
    sample_batch: pygd.Batch,
    batch_norm: bool,
    dropout: float,
) -> None:
    model_class, config = model_config
    config.batch_norm = batch_norm
    config.dropout = dropout
    model = model_class(config)

    model.train()
    output_train = model(sample_batch)

    model.eval()
    output_eval = model(sample_batch)

    if dropout > 0:
        assert not torch.allclose(output_train, output_eval)


def test_self_loops(model_config: ModelConfigTuple, sample_batch: pygd.Batch) -> None:
    model_class, config = model_config

    if not hasattr(config, "add_self_loops"):
        pytest.skip("Self loops test not applicable for GraphSAGE")

    config.add_self_loops = True  # mypy: ignore
    model_with_loops = model_class(config)
    output_with_loops = model_with_loops(sample_batch)

    config.add_self_loops = False
    model_without_loops = model_class(config)
    output_without_loops = model_without_loops(sample_batch)

    assert not torch.allclose(output_with_loops, output_without_loops)


# GAT-specific tests
@pytest.mark.parametrize("heads", [1, 2, 4, 8])
def test_gat_heads(model_config: ModelConfigTuple, sample_batch: pygd.Batch, heads: int) -> None:
    model_class, config = model_config
    if not isinstance(config, GATConfig):
        pytest.skip("Test only applicable to GAT models")

    config.heads = heads
    model = model_class(config)
    output = model(sample_batch)
    assert output is not None


@pytest.mark.parametrize("edge_dim", [0, 1, 4])
def test_gat_edge_features(
    model_config: ModelConfigTuple, sample_batch: pygd.Batch, edge_dim: int
) -> None:
    model_class, config = model_config

    if not isinstance(config, GATConfig):
        pytest.skip("Test only applicable to GAT models")

    config.edge_dim = edge_dim
    if edge_dim > 0:
        num_edges = sample_batch.edge_index.size(1)
        sample_batch.edge_attr = torch.randn(num_edges, edge_dim, device=config.device)

    model = model_class(config)
    output = model(sample_batch)
    assert output is not None
