import pytest
from pydantic import ValidationError

from filtering.core.entities.candidate import InputNodeCandidate
from filtering.core.entities.config.node_filtering import RecencyNodeFilteringConfig
from filtering.core.entities.kg import YearRange
from filtering.impl.kg.networkx_kg import NetworkXKnowledgeGraph, NetworkXKnowledgeGraphConfig
from filtering.impl.node_filtering.recency_filtering import RecencyNodeFiltering


@pytest.fixture
def kg(nodes_path, edges_path):
    return NetworkXKnowledgeGraph(
        NetworkXKnowledgeGraphConfig(nodes_path=nodes_path, edges_path=edges_path)
    )


@pytest.fixture
def test_input_candidates():
    node_ocids = ["102000010821", "102000007446", "102000036354", "102100013400", "238000011830"]
    return [InputNodeCandidate(node_id=ocid) for ocid in node_ocids]


class TestRecencyNodeFilteringConfig:
    def test_init_error_cases(self):
        with pytest.raises(ValidationError):
            RecencyNodeFilteringConfig(year_range=YearRange(2010, 2008))
            RecencyNodeFilteringConfig(year_range=YearRange(2010, 2011), year_window_size=2)


class TestRecencyNodeFiltering:
    def test_filter(self, kg, test_input_candidates):
        m = RecencyNodeFiltering(config=RecencyNodeFilteringConfig(), kg=kg)

        output_cands1 = m.filter(candidates=test_input_candidates)
        output_cands2 = m.filter(candidates=test_input_candidates, max_output_candidates=2)

        assert len(output_cands1) == len(test_input_candidates)
        assert len(output_cands2) == 2
