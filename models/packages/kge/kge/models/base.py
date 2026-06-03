from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
from hakken_ml_toolkit.ml_utils.extras import PyTorchUtils
from hakken_ml_toolkit.ml_utils.extras.scalers import (
    ScalerI,
    SigmoidScaler,
    SigmoidScalerConfig,
)
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset

from kge.common.types import (
    FloatTensor2D,
    LongTensor1D,
    LongTensor2D,
    TensorND,
)
from kge.models.config import KGEConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

    from kge.common.entities import KGEForwardOutput
    from kge.negative_sampler import NegativeSamplerI


T = TypeVar("T", bound=KGEConfig)


KGEType = TypeVar("KGEType", bound="KGEI")


class KGEI(nn.Module, ABC, Generic[T]):
    def __init__(self, config: T):
        self.config = config
        self.sro_batch: LongTensor2D | None

        self._cache_embeddings = False

        self._scaler: ScalerI | None = None

    @property
    def device(self) -> str | torch.device:
        first_param: nn.Parameter = next(self.parameters())
        return first_param.device

    @staticmethod
    def config_file_path(path: str | Path) -> Path:
        return Path(path) / "config.json"

    @staticmethod
    def model_file_path(path: str | Path) -> Path:
        return Path(path) / "model.pt"

    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    def set_cache_embeddings(self, value: bool) -> None:
        self._cache_embeddings = value

    def __call__(self, sro_batch: LongTensor2D) -> KGEForwardOutput:
        return self.forward(sro_batch)

    @abstractmethod
    def eval(self: KGEType) -> KGEType:
        pass

    @abstractmethod
    def train(self: KGEType, mode: bool = True) -> KGEType:
        pass

    @abstractmethod
    def to_device(self: KGEType, device: str | torch.device) -> KGEType:
        pass

    @abstractmethod
    def parameters(self, recurse: bool = True) -> Iterator:
        pass

    @abstractmethod
    def forward(self, sro_batch: LongTensor2D) -> KGEForwardOutput:
        """Forward pass of the model."""
        pass

    @abstractmethod
    def _score_objects(self, s_emb: FloatTensor2D, r_emb: FloatTensor2D) -> FloatTensor2D:
        pass

    def score_objects(self, sr_batch: LongTensor2D) -> FloatTensor2D:
        s_emb = self.entity_embeddings(sr_batch[:, 0])
        r_emb = self.relation_embeddings(sr_batch[:, 1])

        return self._score_objects(s_emb, r_emb)

    @abstractmethod
    def _score_subjects(self, r_emb: FloatTensor2D, o_emb: FloatTensor2D) -> FloatTensor2D:
        pass

    def score_subjects(self, ro_batch: LongTensor2D) -> FloatTensor2D:
        r_emb = self.relation_embeddings(ro_batch[:, 0])
        o_emb = self.entity_embeddings(ro_batch[:, 1])

        return self._score_subjects(r_emb, o_emb)

    @abstractmethod
    def _score_relations(self, s_emb: FloatTensor2D, o_emb: FloatTensor2D) -> FloatTensor2D:
        pass

    def score_relations(self, so_batch: LongTensor2D) -> FloatTensor2D:
        s_emb = self.entity_embeddings(so_batch[:, 0])
        o_emb = self.entity_embeddings(so_batch[:, 1])

        return self._score_relations(s_emb, o_emb)

    @abstractmethod
    def _score(
        self, s_emb: FloatTensor2D, r_emb: FloatTensor2D, o_emb: FloatTensor2D
    ) -> FloatTensor2D:
        pass

    def score(self, sro_batch: LongTensor2D) -> FloatTensor2D:
        s_emb = self.entity_embeddings(sro_batch[:, 0])
        r_emb = self.relation_embeddings(sro_batch[:, 1])
        o_emb = self.entity_embeddings(sro_batch[:, 2])

        return self._score(s_emb, r_emb, o_emb)

    @abstractmethod
    def entity_embeddings(self, entity_batch: LongTensor1D) -> FloatTensor2D:
        pass

    @abstractmethod
    def relation_embeddings(self, relation_batch: LongTensor1D) -> FloatTensor2D:
        pass

    @torch.no_grad()
    def save_embeddings(self, path: Path, device: str | torch.device = "cpu") -> None:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        entity_batch = torch.arange(self.config.num_entities, dtype=torch.long).to(device)
        entity_embeddings = self.entity_embeddings(entity_batch)

        entity_path = path / "entities.pt"

        torch.save(entity_embeddings, entity_path)

        relation_batch = torch.arange(self.config.num_relations, dtype=torch.long).to(device)

        relation_path = path / "relations.pt"

        relation_embeddings = self.relation_embeddings(relation_batch)
        torch.save(relation_embeddings, relation_path)

        logger.info(f"Embeddings saved at {path}")

    @classmethod
    @abstractmethod
    def get_config_class(cls) -> type[T]:
        pass

    def has_scaler(self) -> bool:
        return self._scaler is not None

    @torch.no_grad()
    def normalize_scores(self, scores: FloatTensor2D) -> FloatTensor2D:
        """Normalize scores using the provided scaler."""
        if self._scaler is None:
            msg = "No score scaler set."
            raise ValueError(msg)

        return self._scaler.transform(scores)

    @torch.no_grad()
    def fit_score_scaler_from_dataset(
        self,
        dataset: Dataset,
        negative_sampler: NegativeSamplerI | None = None,
        json_path: str | None = None,
        loader_kwargs: dict | None = None,
    ) -> None:
        if loader_kwargs is None:
            loader_kwargs = {
                "batch_size": 1024,
                "shuffle": True,
                "num_workers": 0,
                "pin_memory": True,
            }

        data_loader = DataLoader(
            dataset=dataset,
            **loader_kwargs,
        )
        self.fit_score_scaler(
            data_loader=data_loader,
            negative_sampler=negative_sampler,
            json_path=json_path,
        )

    def load_score_scaler(self, json_path: str) -> bool:
        success = False
        if Path(json_path).exists():
            logger.info(f"Loading score scaler from {json_path}")
            self._scaler = SigmoidScaler.load(json_path)
            self._scaler.to(self.device)
            success = True
        return success

    @torch.no_grad()
    def fit_score_scaler(
        self,
        data_loader: DataLoader,
        negative_sampler: NegativeSamplerI | None = None,
        json_path: str | None = None,
    ) -> None:
        """Fit a score scaler using batches of triples from the data loader.

        This method samples scores from the model using the provided data loader
        and optionally corrupted negative samples, then fits the scaler to these scores.
        The scaler can then be used to normalize or standardize scores during inference.

        Args:
            scaler: An instance of a scaler that implements the ScalerI interface
                which will be fitted on model scores.
            data_loader: DataLoader containing batches of positive triples to score.
            negative_sampler: Optional negative sampler to generate corrupted triples.
                            If provided, scores from both positive and negative samples
                            will be used to fit the scaler.
            device: The device to run the computation on (CPU or GPU).

        Returns:
            The fitted scaler instance.
        """

        if json_path is not None and self.load_score_scaler(json_path):
            return

        scaler_config = SigmoidScalerConfig(temperature=None, fixed_data_min=[0.0])
        scaler = SigmoidScaler(scaler_config)
        logger.info("Fitting score scaler...")
        scaler.to(self.device)

        if negative_sampler is not None:
            logger.warning(
                "Negative sampler provided will be ignored when fitting the score scaler."
            )
            negative_sampler = None

        if negative_sampler is not None:
            negative_sampler.to_device(self.device)

        self.eval()

        def get_scores_batch() -> Iterator[TensorND]:
            for batch in data_loader:
                sro_batch_pos = batch[0].to(self.device)
                if negative_sampler is not None:
                    sro_tensor_neg = negative_sampler.corrupt_batch(sro_batch=sro_batch_pos)
                    sro_batch_neg = sro_tensor_neg.view(-1, 3)

                    sro_batch = PyTorchUtils.concat_tensors([sro_batch_pos, sro_batch_neg], dim=0)

                else:
                    sro_batch = sro_batch_pos

                scores = self.score(sro_batch)
                yield scores

        scaler.fit_from_iterator(get_scores_batch(), num_batches=10)
        if json_path is not None:
            logger.info(f"Saving score scaler to {json_path}")
            scaler.save(json_path)
        scaler.to(self.device)
        self._scaler = scaler

    def save(self, path: str | Path):
        """path is a directory"""
        state_dict = self.state_dict()
        model_file = self.model_file_path(path)
        config_file = self.config_file_path(path)

        torch.save(state_dict, model_file)

        with open(config_file, "w") as f:
            json.dump(self.config.model_dump(), f)

    @classmethod
    def load_config(cls, model_config_path: Path) -> T:
        config_file_path = cls.config_file_path(model_config_path)
        with open(config_file_path) as f:
            loaded_json = f.read()

        config_class = cls.get_config_class()
        return config_class.model_validate_json(loaded_json)

    @classmethod
    def load(cls: type[KGEType], path: Path, device: str | torch.device = "cpu") -> KGEType:
        config = cls.load_config(path)
        model = cls(config)

        model_file = cls.model_file_path(path)

        state_dict = torch.load(model_file, map_location=device, weights_only=True)

        model.load_state_dict(state_dict)
        return model.to(device)
