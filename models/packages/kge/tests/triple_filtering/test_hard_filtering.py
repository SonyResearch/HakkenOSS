import torch
from hakken_ml_toolkit.ml_base_structures import KnowledgeGraph

from kge.triple_filterer.hard_filtering import (
    HardTripleFilter,
    HardTripleFilterConfig,
)

# Create the facts_dict with training triples
facts_dict = {
    "train": torch.tensor(
        [
            [0, 0, 1],
            [1, 0, 2],
            [2, 0, 3],
        ],
        dtype=torch.long,
    ),
    "val": torch.empty((0, 3), dtype=torch.long),
    "test": torch.empty((0, 3), dtype=torch.long),
}


def test_compute_filtering():
    # Initialize the KnowledgeGraph
    kg = KnowledgeGraph(
        facts_dict=facts_dict,
        num_entities=5,  # Entities: 0 to 4
        num_relations=1,  # Relation: 0
    )
    # Initialize the filterer configuration
    config = HardTripleFilterConfig()
    # Initialize the HardTripleFilter
    filterer = HardTripleFilter(config=config, kg=kg)

    # Prepare a batch of triples (sro_batch)
    sro_batch = torch.tensor(
        [
            [0, 0, 4],  # Test triple not in training set
            [1, 0, 2],  # Triple present in training set
            [2, 0, 4],  # Test triple not in training set
        ],
        dtype=torch.long,
    )

    # Simulate scores for each triple over all entities
    batch_size = sro_batch.size(0)
    scores = torch.rand(batch_size, kg.num_entities)

    # Keep a copy of the original scores for comparison
    original_scores = scores.clone()

    # Apply the compute method
    filtered_scores = filterer.compute_scores(sro_batch, scores)

    assert filtered_scores[0, 1].item() == float("-inf"), (
        "Score for entity 1 should be filtered in the first triple."
    )

    assert torch.equal(filtered_scores[1], original_scores[1]), (
        "Scores should not be modified for the second triple."
    )

    assert filtered_scores[2, 3].item() == float("-inf"), (
        "Score for entity 3 should be filtered in the third triple."
    )

    # Verify that other scores remain unchanged
    assert torch.equal(filtered_scores[0, [0, 2, 3, 4]], original_scores[0, [0, 2, 3, 4]]), (
        "Other scores in the first triple should remain unchanged."
    )
    assert torch.equal(filtered_scores[2, [0, 1, 2, 4]], original_scores[2, [0, 1, 2, 4]]), (
        "Other scores in the third triple should remain unchanged."
    )


def test_no_filtering_needed():
    # Initialize the KnowledgeGraph
    kg = KnowledgeGraph(facts_dict=facts_dict, num_entities=5, num_relations=1)
    config = HardTripleFilterConfig()
    filterer = HardTripleFilter(config=config, kg=kg)

    # Prepare a batch where no filtering should occur
    sro_batch = torch.tensor(
        [
            [0, 0, 1],
            [1, 0, 2],
            [2, 0, 3],
        ],
        dtype=torch.long,
    )

    batch_size = sro_batch.size(0)
    num_entities = 5
    scores = torch.rand(batch_size, num_entities)
    original_scores = scores.clone()

    # Apply the compute method
    filtered_scores = filterer.compute_scores(sro_batch, scores)

    # Since pos_o - {o} is empty for all triples, no scores should change
    assert torch.equal(filtered_scores, original_scores), (
        "Scores should remain unchanged when no filtering is needed."
    )
