from pathlib import Path

from embeddings.core.contracts.graph_loader import IGraphLoader
from embeddings.core.entities.graph import LoadedGraph


class RDFLoader(IGraphLoader):
    def load_from_file(self, file: Path) -> LoadedGraph:
        return LoadedGraph().parse(file)
