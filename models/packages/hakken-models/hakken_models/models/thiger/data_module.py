from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from hakken_models.core.entities.kg_data import KGData
from hakken_models.data_loaders import KGLinkNeighborLoader


class THiGERDataModule(LightningDataModule):
    def __init__(
        self,
        train_kg_data: KGData,
        train_entity_pairs: Tensor,
        train_relations: Tensor,
        val_kg_data: KGData,
        val_entity_pairs: Tensor,
        val_relations: Tensor,
        num_neighbors: list[int],
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        self.train_kg_data = train_kg_data
        self.train_entity_pairs = train_entity_pairs
        self.train_relations = train_relations

        self.val_kg_data = val_kg_data
        self.val_entity_pairs = val_entity_pairs
        self.val_relations = val_relations

        self.num_neighbors = num_neighbors

        self.batch_size = batch_size
        self.kwargs = kwargs

    def train_dataloader(self) -> DataLoader:
        kwargs = self.kwargs.copy()
        kwargs.update({"shuffle": True, "batch_size": self.batch_size})

        return KGLinkNeighborLoader(
            data=self.train_kg_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=self.train_entity_pairs.t().contiguous(),
            edge_label=self.train_relations,
            **kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        kwargs = self.kwargs.copy()

        kwargs.update(
            {
                "shuffle": False,
                "batch_size": self.batch_size,
            }
        )

        return KGLinkNeighborLoader(
            data=self.val_kg_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=self.val_entity_pairs.t().contiguous(),
            edge_label=self.val_relations,
            **kwargs,
        )
