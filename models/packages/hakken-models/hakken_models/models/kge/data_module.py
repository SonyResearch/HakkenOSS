from hakken_ml_toolkit.ml_utils.extras import FactBatchUtils
from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from hakken_models.data_loaders.kge import get_negative_data_loader
from hakken_models.negative_samplers import NegativeSampler


class KGEDataModule(LightningDataModule):
    def __init__(
        self,
        train_facts: Tensor,
        val_facts: Tensor,
        negative_sampler: NegativeSampler,
        val_num_negatives: int = 5_000,
        batch_size: int = 32,
        train_relation_labels: Tensor | None = None,
        val_relation_labels: Tensor | None = None,
        num_relations: int | None = None,
        pair_relation_supervision: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        train_lbl = train_relation_labels
        val_lbl = val_relation_labels

        if pair_relation_supervision and train_relation_labels is None:
            if num_relations is None:
                msg = (
                    "num_relations is required when pair_relation_supervision=True "
                    "without train_relation_labels"
                )
                raise ValueError(msg)
            train_lbl = FactBatchUtils.fact_batch_pair_relation_labels(
                train_facts, num_relations=num_relations
            )

        if pair_relation_supervision and val_relation_labels is None:
            if num_relations is None:
                msg = (
                    "num_relations is required when pair_relation_supervision=True "
                    "without val_relation_labels"
                )
                raise ValueError(msg)
            val_lbl = FactBatchUtils.fact_batch_pair_relation_labels(
                val_facts, num_relations=num_relations
            )

        if train_lbl is not None:
            if train_lbl.shape[0] != train_facts.shape[0]:
                msg = (
                    f"train_relation_labels rows ({train_lbl.shape[0]}) "
                    f"must match train_facts ({train_facts.shape[0]})"
                )
                raise ValueError(msg)
            self.train_dataset = TensorDataset(train_facts, train_lbl)
        else:
            self.train_dataset = TensorDataset(train_facts)

        if val_lbl is not None:
            if val_lbl.shape[0] != val_facts.shape[0]:
                msg = (
                    f"val_relation_labels rows ({val_lbl.shape[0]}) "
                    f"must match val_facts ({val_facts.shape[0]})"
                )
                raise ValueError(msg)
            self.val_dataset = TensorDataset(val_facts, val_lbl)
        else:
            self.val_dataset = TensorDataset(val_facts)

        self.negative_sampler = negative_sampler

        self.batch_size = batch_size
        self.val_num_negatives = val_num_negatives
        self.kwargs = kwargs

    def train_dataloader(self) -> DataLoader:
        kwargs = self.kwargs.copy()
        kwargs.update({"shuffle": True, "batch_size": self.batch_size})

        return get_negative_data_loader(
            self.train_dataset, negative_sampler=self.negative_sampler, **kwargs
        )

    def val_dataloader(self) -> DataLoader:
        kwargs = self.kwargs.copy()

        kwargs.update(
            {
                "shuffle": False,
                "num_negatives": self.val_num_negatives,
                "batch_size": self.batch_size,
            }
        )

        return get_negative_data_loader(
            self.val_dataset, negative_sampler=self.negative_sampler, **kwargs
        )
