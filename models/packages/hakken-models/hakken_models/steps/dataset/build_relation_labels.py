from collections import defaultdict
from typing import Annotated

import numpy as np
import polars as pl
import torch
from loguru import logger
from torch import Tensor
from zenml import ArtifactConfig, log_metadata, step


def build_pair_relation_history(
    facts: Tensor,
) -> dict[tuple[int, int], list[tuple[int, float]]]:
    """Build ``{(s, o): [(r, t), ...]}`` sorted by timestamp ascending."""
    s_list = facts[:, 0].tolist()
    r_list = facts[:, 1].tolist()
    o_list = facts[:, 2].tolist()
    t_list = facts[:, 3].tolist()

    history: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    for s, r, o, t in zip(s_list, r_list, o_list, t_list, strict=False):
        history[(int(s), int(o))].append((int(r), float(t)))

    for v in history.values():
        v.sort(key=lambda x: x[1])

    return dict(history)


def build_fact_relation_labels(
    target_facts: Tensor,
    history: dict[tuple[int, int], list[tuple[int, float]]],
    num_relations: int,
) -> Tensor:
    """Build multi-hot relation labels for each target fact.

    For target fact ``i = (s_i, r_i, o_i, t_i)``, sets
    ``labels[i, r] = 1`` iff:

    * ``r == r_i`` (the target relation itself), **or**
    * ``history`` contains ``(r, t')`` for pair ``(s_i, o_i)`` with
      ``t' < t_i``.

    ``history`` is typically from :func:`build_pair_relation_history` over the
    temporal facts that should count as the knowledge base (e.g. train-only
    for train targets, train+val for val targets).

    Args:
        target_facts: ``[N, 4]`` tensor ``(s, r, o, t)`` — facts to
            produce labels for.
        history: Mapping ``(s, o) -> [(r, t), ...]`` sorted by ``t`` ascending.
        num_relations: Total number of distinct relation types.

    Returns:
        ``[N, num_relations]`` float tensor (multi-hot).
    """
    n = target_facts.shape[0]
    labels = torch.zeros(n, num_relations, dtype=torch.float32)

    s_list = target_facts[:, 0].tolist()
    r_list = target_facts[:, 1].tolist()
    o_list = target_facts[:, 2].tolist()
    t_list = target_facts[:, 3].tolist()

    for i in range(n):
        s_i, r_i, o_i, t_i = int(s_list[i]), int(r_list[i]), int(o_list[i]), t_list[i]
        labels[i, r_i] = 1.0

        for r, t in history.get((s_i, o_i), []):
            if t < t_i:
                labels[i, r] = 1.0

    return labels


@step
def build_relation_labels_step(
    train_np: np.ndarray,
    val_np: np.ndarray,
    relations_map_df: pl.DataFrame,
) -> tuple[
    Annotated[np.ndarray, ArtifactConfig(name="train_relation_labels_np")],
    Annotated[np.ndarray, ArtifactConfig(name="val_relation_labels_np")],
]:
    """Build multi-hot relation labels for train and val splits.

    For each fact ``(s, r, o, t)`` in a split, the label vector
    ``[num_relations]`` has a ``1`` for every relation observed between
    ``(s, o)`` at timestamps strictly before ``t`` in the knowledge
    base, plus a ``1`` for ``r`` itself.

    * **Train** knowledge base = ``train_np``
    * **Val** knowledge base = ``train_np + val_np``

    ``num_relations`` is derived from ``relations_map_df.height``.
    """
    num_relations = relations_map_df.height
    train_facts = torch.from_numpy(train_np).long()
    val_facts = torch.from_numpy(val_np).long()
    all_facts = torch.cat([train_facts, val_facts], dim=0)

    logger.info(
        f"Building relation labels: train={train_facts.shape[0]} facts, "
        f"val={val_facts.shape[0]} facts, num_relations={num_relations}"
    )

    history = build_pair_relation_history(train_facts)

    train_labels = build_fact_relation_labels(
        target_facts=train_facts,
        history=history,
        num_relations=num_relations,
    )

    history = build_pair_relation_history(all_facts)

    val_labels = build_fact_relation_labels(
        target_facts=val_facts,
        history=history,
        num_relations=num_relations,
    )

    train_labels_np = train_labels.numpy()
    val_labels_np = val_labels.numpy()

    log_metadata(
        metadata={
            "train_relation_labels_shape": list(train_labels_np.shape),
            "val_relation_labels_shape": list(val_labels_np.shape),
            "num_relations": num_relations,
            "train_avg_active_relations": float(train_labels_np.sum(axis=1).mean()),
            "val_avg_active_relations": float(val_labels_np.sum(axis=1).mean()),
        },
    )

    logger.info(f"Relation labels built: train={train_labels_np.shape}, val={val_labels_np.shape}")

    return train_labels_np, val_labels_np
