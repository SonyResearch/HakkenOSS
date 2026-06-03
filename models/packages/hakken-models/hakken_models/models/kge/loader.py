from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

from hakken_models.core.configs.train_kge import TrainKGEConfig
from hakken_models.datasets.deployment import DatasetDeployment
from hakken_models.models.kge.base import KGE
from hakken_models.models.loader import ModelLoader


@dataclass(slots=True)
class KGEArtifacts:
    dataset: DatasetDeployment | None
    kge: KGE


class KGELoader(ModelLoader[KGEArtifacts, KGE]):
    def from_params(
        self, params: dict[str, Any], ckpt_path: str, weights_only: bool = False
    ) -> KGEArtifacts:
        config = TrainKGEConfig(**params)

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
            state_dict = {key.removeprefix("kge."): value for key, value in state_dict.items()}

        kge = KGE.from_config(
            config.kge,
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
        )
        kge.load_state_dict(state_dict)
        return KGEArtifacts(dataset=dataset, kge=kge)
