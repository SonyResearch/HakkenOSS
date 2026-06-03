import json
from pathlib import Path
from typing import List

from embeddings.core.entities.embeddings import Embeddings
from embeddings.core.entities.graph import Graph
from pyrdf2vec.samplers import Sampler as RDFSampler
from pyrdf2vec.walkers import Walker as RDFWalker

GRAPH_TRANSFORMER = "transformer"

Walker = RDFWalker
Sampler = RDFSampler


class ITransformer:
    def __init__(self, vector_size: int, type_word2vec: str, walkers: List[Walker]):
        pass

    def save_embeddings_to_path(self, embeddings: Embeddings, entities: List[str], path: Path):
        with open(path, "w") as file:
            entity_embedding_dict = {
                str(entity): embedding.tolist() for entity, embedding in zip(entities, embeddings)
            }
            json.dump(entity_embedding_dict, file, indent=2)

    def transform(self, graph: Graph, entities: List[str]):
        raise NotImplementedError("Method not implemented")
