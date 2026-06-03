import pytest

from filtering.core.entities.candidate import InputTripleCandidate
from filtering.core.entities.config.triple_filtering import RandomTripleFilteringConfig
from filtering.impl.triple_filtering.random_filtering import RandomTripleFiltering


@pytest.fixture
def test_input_candidates():
    return [
        InputTripleCandidate(
            symbol_mappings=[
                {"variable": f"v_{i}_0", "key": "k_0", "description": f"v_{i}_0"},
                {"variable": f"v_{i}_1", "key": "k_1", "description": f"v_{i}_1"},
            ],
            triple={"subject": f"v_{i}_0", "relation": f"r_{i}", "object": f"v_{i}_0"},
        )
        for i in range(100)
    ]


class TestRandomTripleFilteringModel:
    def test_filter(self, test_input_candidates):
        m1 = RandomTripleFiltering(RandomTripleFilteringConfig(random_seed=100))
        m2 = RandomTripleFiltering(RandomTripleFilteringConfig(random_seed=100))
        m3 = RandomTripleFiltering(RandomTripleFilteringConfig(random_seed=200))

        output_cands1 = m1.filter(candidates=test_input_candidates, max_output_candidates=10)
        output_cands2 = m2.filter(candidates=test_input_candidates, max_output_candidates=10)
        output_cands3 = m3.filter(candidates=test_input_candidates, max_output_candidates=10)
        assert output_cands1 == output_cands2
        assert output_cands1 != output_cands3
        assert len(output_cands1) == len(output_cands2) == len(output_cands3)
        for oc in output_cands1:
            assert oc.filter_score == -1
