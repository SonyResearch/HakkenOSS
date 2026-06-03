"""Loader for trained SeGAL models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from langchain_core.embeddings import Embeddings
from loguru import logger

from hakken_models.core.embedder import create_embedder
from hakken_models.datasets.deployment import DatasetDeployment
from hakken_models.models.loader import ModelLoader
from hakken_models.models.segal.base import SeGAL
from hakken_models.models.segal.inference import SeGALInferenceWrapper


@dataclass
class SeGALArtifacts:
    """Artifacts returned when loading a trained SeGAL model."""

    dataset: DatasetDeployment | None
    segal: SeGAL
    _embedder: Embeddings | None = field(default=None, repr=False)

    @property
    def embedder(self) -> Embeddings:
        """Lazy-loaded embedder from the model config."""
        if self._embedder is None:
            self._embedder = create_embedder(self.segal.config.embedder)
        return self._embedder

    def to_inference_wrapper(self) -> SeGALInferenceWrapper:
        """Build a deployment-ready wrapper with embeddings from the dataset."""
        if self.dataset is None:
            raise ValueError("Dataset is required to build inference wrapper.")
        return SeGALInferenceWrapper(
            segal=self.segal,
            node_embeddings=self.dataset.get_node_embedding_matrix(),
            relation_embeddings=self.dataset.get_relation_embedding_matrix(),
        )


class SeGALLoader(ModelLoader[SeGALArtifacts, SeGAL]):
    """Load a trained SeGAL model from MLflow or directory.

    Returns (SeGAL, DatasetDeployment). Embeddings are loaded from the dataset
    at inference time via dataset.get_node_embedding_matrix() and
    dataset.get_relation_embedding_matrix().
    """

    def from_params(
        self, params: dict[str, Any], ckpt_path: str, weights_only: bool = False
    ) -> SeGALArtifacts:
        from hakken_models.core.configs.train_segal import TrainSeGALConfig

        config = TrainSeGALConfig(**params)

        target_root = self.config.data_root_uri_template.format(
            name=config.dataset.name,
            version=config.dataset.version,
        )

        dataset = DatasetDeployment(target_root=target_root)

        logger.info(f"Loading {ckpt_path}")

        ckpt_dict = torch.load(
            ckpt_path, map_location=self.config.device, weights_only=weights_only
        )

        state_dict: dict[str, Any] = ckpt_dict["state_dict"]

        if self.config.ckpt_is_lightning:
            state_dict = {
                key.removeprefix("segal."): value
                for key, value in state_dict.items()
                if key.startswith("segal.")
            }

        segal = SeGAL(config=config.segal)
        segal.load_state_dict(state_dict, strict=False)

        return SeGALArtifacts(dataset=dataset, segal=segal)
