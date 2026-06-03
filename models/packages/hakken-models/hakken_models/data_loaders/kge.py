from functools import partial
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from hakken_models.negative_samplers import NegativeSampler


def facts_negative_collate(
    batch: list[tuple[Tensor, ...]],
    negative_sampler: NegativeSampler,
    num_negatives: int,
    num_attempts: int = 1,
):
    """Collate positives and generate negatives on-the-fly.

    Args:
        batch: Each element is either ``(fact,)`` with fact ``[>=3]`` (S,R,O,…), or
            ``(fact, relation_labels)`` with ``relation_labels`` ``[num_relations]``
            multi-hot aligned to that fact.
        negative_sampler: Sampler used to corrupt facts.
        num_negatives: Number of negatives per positive.
        num_attempts: Retry attempts if validator rejects (default: 1).

    Returns:
        Dict with ``positives`` ``[B, >=3]``, ``negatives`` ``[B, K, >=3]``, and
        optionally ``relation_labels`` ``[B, R]``.
    """

    first = batch[0]
    if len(first) >= 2:
        positives = torch.stack([item[0] for item in batch])
        relation_labels = torch.stack([item[1] for item in batch])
    else:
        positives = torch.stack([item[0] for item in batch])
        relation_labels = None

    negatives = negative_sampler.corrupt_facts(
        facts=positives, num_negatives=num_negatives, num_attempts=num_attempts
    )

    out: dict[str, Tensor] = {"positives": positives, "negatives": negatives}
    if relation_labels is not None:
        out["relation_labels"] = relation_labels
    return out


def get_negative_data_loader(
    dataset: Dataset,
    negative_sampler: NegativeSampler,
    num_negatives: int = 32,
    num_attempts: int = 1,
    batch_size: int = 256,
    **kwargs: Any,
) -> DataLoader:
    """Create DataLoader with on-the-fly negative sampling.

    Args:
        dataset: Dataset yielding positive triples [3].
        negative_sampler: Sampler for generating negatives.
        num_negatives: Negatives per positive (default: 32).
        num_attempts: Retry attempts for valid negatives (default: 1).
        batch_size: Batch size (default: 256).
        **kwargs: Passed to DataLoader (shuffle, num_workers, etc.).

    Returns:
        DataLoader yielding {"positives": ..., "negatives": ...}.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(
            facts_negative_collate,
            negative_sampler=negative_sampler,
            num_negatives=num_negatives,
            num_attempts=num_attempts,
        ),
        **kwargs,
    )
