import logging
from pathlib import Path
from typing import Protocol

from dependency_injector.wiring import Provide, inject
from pydantic import BaseModel

from embeddings.core.contracts.embeddings import GRAPH_TRANSFORMER, ITransformer
from embeddings.core.contracts.graph_converter import GRAPH_CONVERTER, IGraphConverter
from embeddings.core.contracts.graph_loader import GRAPH_LOADER, IGraphLoader


class ExtractEmbeddingsActionInput(BaseModel):
    ontology_file_path: Path
    output_file_path: Path
    entities_file_path: Path


class ExtractEmbeddingsAction(Protocol):
    @inject
    @staticmethod
    def run(
        input: ExtractEmbeddingsActionInput,
        graph_loader: IGraphLoader = Provide[GRAPH_LOADER],
        graph_converter: IGraphConverter = Provide[GRAPH_CONVERTER],
        transformer: ITransformer = Provide[GRAPH_TRANSFORMER],
    ) -> None:
        logging.info(f"Parsing graph from {input.ontology_file_path}")
        loaded_graph = graph_loader.load_from_file(input.ontology_file_path)

        logging.info("Converting graph")
        graph = graph_converter.convert_graph(loaded_graph)

        entities = [line.strip() for line in open(input.entities_file_path).readlines()]

        logging.info("Calculating embeddings")
        embeddings = transformer.transform(graph, entities)
        transformer.save_embeddings_to_path(embeddings, entities, input.output_file_path)
