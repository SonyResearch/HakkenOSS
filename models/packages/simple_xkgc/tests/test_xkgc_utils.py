from pathlib import Path
from unittest.mock import Mock

import networkx as nx
import pytest
import torch
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph

from hakken_explainer.utils import ExplainerUtils


@pytest.fixture
def mock_kg() -> KnowledgeGraph:
    """Create a mock KnowledgeGraph for testing."""
    kg = Mock(spec=KnowledgeGraph)

    kg.entity_mapping.index_to_id = {0: "entity1", 1: "entity2", 2: "entity3"}
    kg.relation_mapping.index_to_id = {0: "rel1", 1: "rel2"}

    facts_tensor1 = torch.tensor([[0, 0, 1], [1, 1, 2]])
    facts_tensor2 = torch.tensor([[0, 1, 2]])

    kg.facts_dict = {"train": facts_tensor1, "test": facts_tensor2}

    return kg


@pytest.fixture
def tmp_graph_file(tmp_path: Path) -> Path:
    """Create a temporary file path for graph pickle tests."""
    return tmp_path / "test_graph.pkl"


def test_save_and_load_graph(tmp_graph_file: Path) -> None:
    """Test saving and loading a graph pickle."""
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    graph.add_edge("A", "B", key="test")

    ExplainerUtils.save_graph(graph, tmp_graph_file)

    assert tmp_graph_file.exists()

    loaded_graph = ExplainerUtils.load_graph(tmp_graph_file)

    assert isinstance(loaded_graph, nx.MultiDiGraph)
    assert loaded_graph.has_edge("A", "B", key="test")
