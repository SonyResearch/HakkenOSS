import pytest

from filtering.core.entities.kg import EdgeDirection, YearRange
from filtering.impl.kg.utils.edge_cache import EdgeCache


class TestEdgeCache:
    def test_from_pickle(self, edge_cache_pickle_directory):
        EdgeCache.from_pickle_directory(edge_cache_pickle_directory)

    def test_from_ndjson(self, edge_cache_ndjson_directory):
        EdgeCache.from_ndjson_directory(edge_cache_ndjson_directory)

    def test_invalid_path(self):
        with pytest.raises(ValueError):
            EdgeCache.from_pickle_directory(directory="_INCORRECT_PATH_")
        with pytest.raises(ValueError):
            EdgeCache.from_ndjson_directory(directory="_INCORRECT_PATH_")

    @pytest.mark.parametrize("direction", [EdgeDirection.IN, EdgeDirection.OUT])
    def test_has_node_id(self, direction):
        edge_cache = EdgeCache()
        edge_cache.add_edge(node_id="node_id1", year=2010, direction=direction)

        wrong_direction = EdgeDirection.OUT if direction == EdgeDirection.IN else EdgeDirection.IN

        assert edge_cache.has_node_id("node_id1", direction=direction)
        assert not edge_cache.has_node_id("node_id1", direction=wrong_direction)
        assert not edge_cache.has_node_id("_NOT_EXIST_", direction=direction)

    @pytest.mark.parametrize("direction", [EdgeDirection.IN, EdgeDirection.OUT])
    @pytest.mark.parametrize("year_range", [None, YearRange(2010, 2020)])
    def test_get_degree(self, direction, year_range):
        edge_cache = EdgeCache()
        edge_cache.add_edge(node_id="node_id1", year=2010, direction=direction)
        edge_cache.add_edge(node_id="node_id1", year=2011, direction=direction)
        edge_cache.add_edge(node_id="node_id1", year=2012, direction=direction)
        edge_cache.add_edge(node_id="node_id1", year=2020, direction=direction)
        edge_cache.add_edge(node_id="node_id1", year=2021, direction=direction)

        wrong_direction = EdgeDirection.OUT if direction == EdgeDirection.IN else EdgeDirection.IN

        if year_range is None:
            assert (
                edge_cache.get_degree(
                    node_id="node_id1", direction=direction, year_range=year_range
                )
                == 5
            )
        else:
            assert (
                edge_cache.get_degree(
                    node_id="node_id1", direction=direction, year_range=year_range
                )
                == 3
            )
        assert (
            edge_cache.get_degree(
                node_id="node_id1", direction=wrong_direction, year_range=year_range
            )
            == 0
        )
        assert (
            edge_cache.get_degree(node_id="_NOT_EXIST_", direction=direction, year_range=year_range)
            == 0
        )
        assert (
            edge_cache.get_degree(
                node_id="_NOT_EXIST_", direction=wrong_direction, year_range=year_range
            )
            == 0
        )
