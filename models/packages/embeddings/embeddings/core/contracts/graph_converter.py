from typing import Protocol

from embeddings.core.entities.graph import Graph, LoadedGraph

GRAPH_CONVERTER = "graph_converter"


class IGraphConverter(Protocol):
    def convert_graph(self, graph: LoadedGraph) -> Graph:
        raise NotImplementedError()
