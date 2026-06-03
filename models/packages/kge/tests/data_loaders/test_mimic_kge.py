from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

from kge.common.entities import KGData, KGSubgraph
from kge.data_loaders.mimic_kge import BatchType, MimicKGEDataLoader
from kge.data_loaders.mimic_kge.config import MimicKGEDataLoaderConfig
from kge.models.kge_api import KGEAPI


@pytest.fixture
def mock_data() -> Data:
    return Data(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_type=torch.tensor([0, 1, 0], dtype=torch.long),
        num_nodes=4,
        n_id=torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )


@pytest.fixture
def mock_kge() -> MagicMock:
    kge = MagicMock(spec=KGEAPI)
    kge.score.side_effect = lambda x: torch.ones(x.size(0))  # Return scores matching input size
    return kge


@pytest.fixture
def config() -> MimicKGEDataLoaderConfig:
    return MimicKGEDataLoaderConfig(
        num_relations=2,
        num_neighbors=[2, 2],
        batch_size=2,
        num_batches_for_scaling=2,
        negs_per_pos=1,
        corrupt_probs=(0.33, 0.33, 0.34),
        shuffle=True,
    )


def test_initialization(mock_data: Data, mock_kge: MagicMock) -> None:
    loader = MimicKGEDataLoader(
        data=mock_data,
        trained_kge=mock_kge,
        num_relations=2,
        num_neighbors=[2, 2],
        batch_size=2,
    )
    assert loader.num_relations == 2
    assert loader.batch_size == 2
    assert loader.trained_kge == mock_kge
    assert not loader._is_scaler_fitted


def test_to_facts_without_node_ids() -> None:
    loader = MimicKGEDataLoader(
        data=Data(edge_index=torch.tensor([[0, 1], [1, 2]]), edge_type=torch.tensor([0, 1])),
        trained_kge=MagicMock(spec=KGEAPI),
        num_relations=2,
        num_neighbors=[2],
        batch_size=2,
    )
    sub_ids = torch.tensor([0, 1])
    rel_ids = torch.tensor([0, 1])
    obj_ids = torch.tensor([1, 2])
    facts = loader.to_facts(sub_ids, rel_ids, obj_ids)
    expected = torch.tensor([[0, 0, 1], [1, 1, 2]])
    assert torch.equal(facts, expected)


def test_to_facts_with_node_ids() -> None:
    loader = MimicKGEDataLoader(
        data=Data(edge_index=torch.tensor([[0, 1], [1, 2]]), edge_type=torch.tensor([0, 1])),
        trained_kge=MagicMock(spec=KGEAPI),
        num_relations=2,
        num_neighbors=[2],
        batch_size=2,
    )
    sub_ids = torch.tensor([0, 1])
    rel_ids = torch.tensor([0, 1])
    obj_ids = torch.tensor([1, 2])
    node_ids = torch.tensor([10, 20, 30])
    facts = loader.to_facts(sub_ids, rel_ids, obj_ids, node_ids)
    expected = torch.tensor([[10, 0, 20], [20, 1, 30]])
    assert torch.equal(facts, expected)


@patch("os.path.exists", return_value=False)
def test_load_scaler_file_not_exists(
    _mock_exists: MagicMock, mock_data: Data, mock_kge: MagicMock
) -> None:
    loader = MimicKGEDataLoader(
        data=mock_data,
        trained_kge=mock_kge,
        num_relations=2,
        num_neighbors=[2],
        batch_size=2,
        scaler_path="nonexistent.pth",
    )
    assert not loader.load_scaler()
    assert not loader._is_scaler_fitted


def test_collate_fn(mock_data: Data, mock_kge: MagicMock) -> None:
    loader = MimicKGEDataLoader(
        data=mock_data,
        trained_kge=mock_kge,
        num_relations=2,
        num_neighbors=[2],
        batch_size=2,
        negs_per_pos=1,
    )
    index: list[int] = [0, 1]
    with (
        patch(
            "kge.negative_sampler.simple.negative_sampler",
            return_value=(
                torch.tensor([0, 1]),
                torch.tensor([1, 0]),
                torch.tensor([0, 1]),
            ),
        ) as _mock_negative_sampler,
        patch(
            "torch_geometric.loader.LinkNeighborLoader.collate_fn",
            return_value=Data(
                n_id=torch.tensor([0, 1, 2, 3]),
                edge_label_index=torch.tensor([[0, 1], [1, 2]]),
                edge_label=torch.tensor([0, 1]),
            ),
        ) as _mock_collate_fn,
    ):
        batch, facts_pos, facts_neg, scores_pos, scores_neg = loader.collate_fn(index)

    assert isinstance(batch, Data)
    assert hasattr(batch, "node_ids")
    assert hasattr(batch, "edge_label_index")
    assert hasattr(batch, "edge_label")
    assert torch.equal(batch.node_ids, torch.tensor([0, 1, 2, 3]))
    assert facts_pos.shape == (2, 3)
    assert facts_neg.shape == (2, 3)
    assert scores_pos.shape == torch.Size([2, 1])
    assert scores_neg.shape == torch.Size([2, 1])


def test_from_config(
    mock_data: Data, mock_kge: MagicMock, config: MimicKGEDataLoaderConfig
) -> None:
    loader = MimicKGEDataLoader.from_config(config, mock_data, mock_kge)
    assert loader.num_relations == config.num_relations
    assert loader.batch_size == config.batch_size
    assert loader.trained_kge == mock_kge
    assert loader.negs_per_pos == config.negs_per_pos


def test_non_contiguous_facts(mock_kge: MagicMock) -> None:
    num_entities = 6
    num_relations = 2
    train_facts = torch.tensor(
        [
            [1, 0, 2],
            [2, 1, 3],
            [3, 1, 5],
        ],
        dtype=torch.long,
    )

    train_data = KGData.from_facts(
        train_facts,
        num_nodes=num_entities,
        num_relations=num_relations,
    )

    loader = MimicKGEDataLoader(
        data=train_data,
        trained_kge=mock_kge,
        num_relations=num_relations,
        num_neighbors=[2],
        edge_label_index=train_data.edge_index,
        edge_label=train_data.edge_type,
        batch_size=3,
        negs_per_pos=0,
        shuffle=False,
    )

    batch_list: BatchType
    for batch_list in iter(loader):
        assert len(batch_list) == 5

        subgraph, facts_pos, facts_neg, _scores_pos, scores_neg = batch_list

        assert facts_neg is None
        assert scores_neg is None
        assert facts_pos.shape == (3, 3)

        edges_intex_from_facts = facts_pos[:, [0, 2]].t()

        assert torch.all(edges_intex_from_facts == subgraph.edge_index)
        train_data_2 = KGSubgraph.to_kg_data(subgraph)
        assert torch.all(train_data_2.edge_index == train_data.edge_index)
        assert not torch.all(edges_intex_from_facts == train_data.edge_index)


def test_train_valid_facts(mock_kge: MagicMock) -> None:
    num_entities = 4
    num_relations = 2
    train_facts = torch.tensor(
        [
            [0, 0, 1],
            [1, 1, 2],
            [2, 1, 3],
        ],
        dtype=torch.long,
    )
    valid_facts = torch.tensor(
        [
            [0, 0, 2],
            [1, 1, 3],
        ],
        dtype=torch.long,
    )
    train_data = KGData.from_facts(
        train_facts,
        num_nodes=num_entities,
        num_relations=num_relations,
    )

    valid_data = KGData.from_facts(
        valid_facts,
        num_nodes=num_entities,
        num_relations=num_relations,
    )

    loader = MimicKGEDataLoader(
        data=train_data,
        trained_kge=mock_kge,
        num_relations=num_relations,
        num_neighbors=[2],
        edge_label_index=valid_data.edge_index,
        edge_label=valid_data.edge_type,
        batch_size=2,
        negs_per_pos=0,
        shuffle=False,
    )

    for batch_list in iter(loader):
        assert len(batch_list) == 5

        graph, facts_pos, facts_neg, _scores_pos, scores_neg = batch_list

        assert facts_neg is None
        assert scores_neg is None
        assert torch.all(graph.edge_label_index == valid_data.edge_index)

        assert torch.all(facts_pos == valid_facts)
