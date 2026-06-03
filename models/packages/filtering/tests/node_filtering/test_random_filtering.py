import pytest

from filtering.core.entities.candidate import InputNodeCandidate
from filtering.core.entities.config.node_filtering import RandomNodeFilteringConfig
from filtering.impl.node_filtering.random_filtering import RandomNodeFiltering


@pytest.fixture
def test_input_candidates():
    return [InputNodeCandidate(node_id=f"node_{i}") for i in range(100)]


class TestRandomNodeFiltering:
    def test_filter(self, test_input_candidates):
        m1 = RandomNodeFiltering(RandomNodeFilteringConfig(random_seed=100))
        m2 = RandomNodeFiltering(RandomNodeFilteringConfig(random_seed=100))
        m3 = RandomNodeFiltering(RandomNodeFilteringConfig(random_seed=200))

        output_cands1 = m1.filter(candidates=test_input_candidates, max_output_candidates=10)
        output_cands2 = m2.filter(candidates=test_input_candidates, max_output_candidates=10)
        output_cands3 = m3.filter(candidates=test_input_candidates, max_output_candidates=10)
        assert output_cands1 == output_cands2
        assert output_cands1 != output_cands3
        assert len(output_cands1) == len(output_cands2) == len(output_cands3)
