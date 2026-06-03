"""Tests for :class:`KGEDataModule` pair relation supervision."""

import pytest
import torch
from hakken_ml_toolkit.ml_utils.extras import FactBatchUtils

from hakken_models.models.kge.data_module import KGEDataModule
from hakken_models.negative_samplers import UniformNegativeSampler


def test_fact_batch_pair_relation_labels_empty() -> None:
    facts = torch.zeros(0, 4, dtype=torch.long)
    out = FactBatchUtils.fact_batch_pair_relation_labels(facts, num_relations=5)
    assert out.shape == (0, 5)
    assert out.dtype == torch.float32


def test_fact_batch_pair_relation_labels_same_pair_same_row() -> None:
    """Two facts with same (s, o) get identical multi-hot rows."""
    # (s, r, o) — two relations on same pair
    facts = torch.tensor(
        [
            [0, 0, 1, 0],
            [0, 2, 1, 1],
        ],
        dtype=torch.long,
    )
    num_relations = 5
    labels = FactBatchUtils.fact_batch_pair_relation_labels(facts, num_relations=num_relations)
    assert labels.shape == (2, num_relations)
    torch.testing.assert_close(labels[0], labels[1])
    assert labels[0, 0] == 1.0
    assert labels[0, 2] == 1.0


def test_kge_data_module_pair_supervision_requires_num_relations() -> None:
    train = torch.tensor([[0, 0, 1]], dtype=torch.long)
    val = torch.tensor([[1, 1, 2]], dtype=torch.long)
    neg = UniformNegativeSampler(num_entities=10, num_relations=5)
    with pytest.raises(ValueError, match="num_relations"):
        KGEDataModule(
            train_facts=train,
            val_facts=val,
            negative_sampler=neg,
        )


def test_kge_data_module_pair_supervision_batch_has_relation_labels() -> None:
    train = torch.tensor([[0, 0, 1], [0, 2, 1]], dtype=torch.long)
    val = torch.tensor([[1, 1, 2]], dtype=torch.long)
    neg = UniformNegativeSampler(num_entities=10, num_relations=5)
    dm = KGEDataModule(
        train_facts=train,
        val_facts=val,
        negative_sampler=neg,
        batch_size=2,
        num_relations=5,
    )
    batch = next(iter(dm.train_dataloader()))
    assert "relation_labels" in batch
    assert batch["relation_labels"].shape == (2, 5)
    assert batch["positives"].shape[0] == 2


def test_kge_data_module_explicit_labels_override_pair() -> None:
    train = torch.tensor([[0, 0, 1]], dtype=torch.long)
    val = torch.tensor([[1, 1, 2]], dtype=torch.long)
    explicit = torch.zeros(1, 5, dtype=torch.float32)
    explicit[0, 3] = 1.0
    neg = UniformNegativeSampler(num_entities=10, num_relations=5)
    dm = KGEDataModule(
        train_facts=train,
        val_facts=val,
        negative_sampler=neg,
        batch_size=1,
        num_relations=5,
        train_relation_labels=explicit,
        val_relation_labels=torch.zeros(1, 5, dtype=torch.float32),
    )
    batch = next(iter(dm.train_dataloader()))
    torch.testing.assert_close(batch["relation_labels"][0], explicit[0])
