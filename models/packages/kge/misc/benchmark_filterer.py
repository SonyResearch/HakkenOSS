from loguru import logger
import click


from hakken_ml_toolkit.ml_base_structures.data_generator import DummyDataGenerator


from dataclasses import dataclass
from typing import Protocol

import torch
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph
from hakken_ml_toolkit.ml_utils import TimersCollection
from pykeen.sampling.filtering import Filterer, PythonSetFilterer

from kge.common.types import FloatTensor2D, LongTensor2D
from kge.triple_filterer.hard_filtering import (
    HardTripleFilter,
    HardTripleFilterConfig,
)


@click.command()
@click.option("--num_entities", type=int, default=100, help="Number of entities")
@click.option("--num_relations", type=int, default=10, help="Number of relations")
@click.option("--num_triples", type=int, default=1024, help="Number of triples")
@click.option(
    "--device", type=str, default="cpu", help='Device to use (e.g., "cpu" or "cuda")'
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option(
    "--dataset_name", type=str, default="Dummy Dataset", help="Name of the dataset"
)
@dataclass
class FiltererBenchmarkingInput:
    kg: KnowledgeGraph
    sro_batch: LongTensor2D
    scores: FloatTensor2D


@dataclass
class FiltererBenchmarkingOutput:
    time_pykeen: float
    time_custom: float


class FiltererBenchmarking(Protocol):
    """
    Protocol for benchmarking different filtering approaches in knowledge graph
    embedding models.

    This class compares the performance of two triple filtering implementations:
    1. A custom HardTripleFilter implementation
    2. PyKEEN's PythonSetFilterer implementation

    Triple filtering is commonly used in link prediction tasks to ensure that
    existing triples in the knowledge graph are not considered as negative
    examples during evaluation, which would unfairly penalize the model.

    The benchmarking measures execution time for both approaches and verifies
    that they produce identical results. This helps in identifying the most
    efficient filtering method for large-scale knowledge graph embedding applications.

    Methods:
        alternative_approach: Implementation of the PyKEEN-based filtering approach.
        run: Executes the benchmarking comparison between the two filtering methods.
    """

    @staticmethod
    def alternative_approach(
        sro_batch: torch.Tensor, scores: torch.Tensor, filterer: Filterer
    ) -> torch.Tensor:
        # get batch size and num_entities
        batch_size, num_entities = scores.shape

        # Prepare all possible triples for the batch
        sro_all_batch = sro_batch.repeat_interleave(num_entities, dim=0)
        object_range = torch.arange(
            num_entities, device=sro_batch.device, dtype=torch.long
        ).repeat(batch_size)
        sro_all_batch[:, 2] = object_range

        # Check which triples are in the training data
        is_contained = filterer.contains(sro_all_batch)

        # Reshape the boolean mask to [batch_size, num_entities]
        is_contained_resized = is_contained.view(batch_size, num_entities)

        # Drop the evaluating triplet
        is_contained_resized[
            torch.arange(is_contained_resized.size(0)), sro_batch[:, -1].squeeze()
        ] = False

        # Filter out the scores for existing triples
        scores[is_contained_resized] = float("-inf")

        return scores

    @staticmethod
    def run(inputs: FiltererBenchmarkingInput) -> FiltererBenchmarkingOutput:
        """
        Runs the benchmarking of two filtering approaches and prints their execution times.
        """

        kg = inputs.kg
        scores = inputs.scores
        sro_batch = inputs.sro_batch
        # Approach A: Using HardTripleFilter

        timer = TimersCollection()

        timer.tic("time/hard_filtering")
        config = HardTripleFilterConfig()
        filterer = HardTripleFilter(config=config, kg=kg)
        scores_a = filterer.compute_scores(sro_batch=sro_batch, scores=scores.clone())
        elapsed_time_custom = timer.toc("time/hard_filtering")

        # Approach B: Using alternative approach with PythonSetFilterer
        timer.tic("time/hard_filtering_pykeen")
        my_filter = PythonSetFilterer(mapped_triples=kg.facts_dict["train"].data)
        scores_b = FiltererBenchmarking.alternative_approach(
            sro_batch=sro_batch, scores=scores.clone(), filterer=my_filter
        )
        elapsed_time_pykeen = timer.toc("time/hard_filtering_pykeen")
        if torch.equal(scores_a, scores_b):
            msg = "The scores from both approaches do not match."
            raise ValueError(msg)

        return FiltererBenchmarkingOutput(
            time_pykeen=elapsed_time_pykeen, time_custom=elapsed_time_custom
        )


def main(
    num_entities: int,
    num_relations: int,
    num_triples: int,
    device: str,
    seed: int,
    dataset_name: str,
):
    kg = DummyDataGenerator.knowledge_graph(
        batch_size=num_triples,
        num_entities=num_entities,
        num_relations=num_relations,
        device=device,
        seed=seed,
    )

    sro_batch = kg.facts_dict["all"]

    scores = DummyDataGenerator.scores(
        batch_size=num_triples, num_entities=num_entities
    )
    inputs = FiltererBenchmarkingInput(kg=kg, sro_batch=sro_batch, scores=scores)
    output = FiltererBenchmarking.run(inputs=inputs, dataset_name=dataset_name)

    logger.info(
        f"Pykeen PythonSetFilterer execution time {dataset_name}: {output.time_pykeen:.4f} seconds"
    )
    logger.info(
        f"Homemade PythonSetFilterer execution time {dataset_name}:"
        f" {output.time_homemade:.4f} seconds"
    )


if __name__ == "__main__":
    main()
