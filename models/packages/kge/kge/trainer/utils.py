import gc
from typing import Protocol, cast

import torch
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset


class DataLoaderFactory:
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, **dataloader_kwargs):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        dataloader_kwargs.pop("batch_size", None)
        dataloader_kwargs.pop("_target_", None)
        self.dataloader_kwargs = dataloader_kwargs

    def create_train_loader(self, batch_size: int):
        return DataLoader(self.train_dataset, batch_size=batch_size, **self.dataloader_kwargs)

    def create_val_loader(self, batch_size: int):
        return DataLoader(self.val_dataset, batch_size=batch_size, **self.dataloader_kwargs)


class TrainUtils(Protocol):
    @staticmethod
    def get_dataloaders(
        model_pl: LightningModule,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        batch_size_optimization_config: DictConfig,
        loader_config: DictConfig,
        trainer_config: DictConfig,
    ) -> list[DataLoader]:
        if batch_size_optimization_config.enable:
            optimal_batch_size = TrainUtils.find_max_batch_size(
                module=model_pl,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                trainer_config=cast("dict", trainer_config),
                loader_config=cast("dict", loader_config),
                starting_batch_size=batch_size_optimization_config.starting_batch_size,
                max_batch_size=batch_size_optimization_config.max_batch_size,
            )

            logger.info(f"Optimal batch size: {optimal_batch_size}")

            loader_config["batch_size"] = optimal_batch_size
        train_dataloader = DataLoader(train_dataset, shuffle=True, **loader_config)

        valid_dataloader = DataLoader(valid_dataset, shuffle=False, **loader_config)
        return [train_dataloader, valid_dataloader]

    @staticmethod
    def find_max_batch_size(
        module: LightningModule,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        trainer_config: dict,
        loader_config: dict,
        starting_batch_size: int = 32,
        max_batch_size: int = 16384,
    ):
        if starting_batch_size > max_batch_size:
            return max_batch_size

        batch_size = starting_batch_size
        dl_factory = DataLoaderFactory(
            train_dataset=train_dataset, val_dataset=valid_dataset, **loader_config
        )
        test_trainer_config = trainer_config.copy()

        test_trainer_config["limit_train_batches"] = 2
        test_trainer_config["limit_val_batches"] = 2
        test_trainer_config["max_epochs"] = 1
        test_trainer_config["enable_checkpointing"] = False

        optimal_batch_size = batch_size
        should_continue = True
        while should_continue:
            logger.info(f"Trying batch_size={batch_size} (max={max_batch_size})")

            try:
                train_loader = dl_factory.create_train_loader(batch_size)
                val_loader = dl_factory.create_val_loader(batch_size)

                trainer = Trainer(**test_trainer_config)
                trainer.fit(module, train_loader, val_loader)
                if batch_size >= max_batch_size:
                    logger.info(
                        f"Reached the configured max ({max_batch_size}). Using {batch_size}."
                    )
                    optimal_batch_size = batch_size
                    should_continue = False

                # If successful, try larger batch size
                batch_size *= 2

            except torch.OutOfMemoryError:
                logger.info(f"Batch size {batch_size} failed. Backing off to {batch_size // 2}.")
                optimal_batch_size = batch_size // 2
                should_continue = False
            finally:
                del trainer, train_loader, val_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        return optimal_batch_size
