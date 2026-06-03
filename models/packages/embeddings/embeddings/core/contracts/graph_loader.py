from pathlib import Path
from typing import Protocol

from embeddings.core.entities.graph import LoadedGraph

GRAPH_LOADER = "graph_loader"


class IGraphLoader(Protocol):
    def load_from_file(self, file: Path) -> LoadedGraph:
        raise NotImplementedError()
