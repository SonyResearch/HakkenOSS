from typing import List

from embeddings.core.contracts.embeddings import ITransformer, Walker
from embeddings.core.entities.embeddings import Embeddings
from embeddings.core.entities.graph import Graph
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.rdf2vec import RDF2VecTransformer


def get_sg_value(type_word2vec: str):
    if type_word2vec == "CBOW":
        return 0
    elif type_word2vec == "skip-gram":
        return 1
    return None


class Transformer(ITransformer):
    def __init__(self, vector_size: int, type_word2vec: str, walkers: List[Walker]):
        super().__init__(vector_size, type_word2vec, walkers)
        self.transformer = RDF2VecTransformer(
            Word2Vec(vector_size=vector_size, sg=get_sg_value(type_word2vec)),
            walkers=walkers,
            verbose=1,
        )

    def transform(self, graph: Graph, entities: List[str]) -> Embeddings:
        embeddings, _ = self.transformer.fit_transform(graph, entities)
        return embeddings
