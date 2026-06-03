from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

from hakken_models.core.configs.train_thiger import TrainTHiGERConfig
from hakken_models.datasets.deployment import DatasetDeployment
from hakken_models.models.loader import ModelLoader
from hakken_models.models.thiger.base import THiGER


@dataclass(slots=True)
class THiGERArtifacts:
    dataset: DatasetDeployment | None
    thiger: THiGER


class THiGERLoader(ModelLoader[THiGERArtifacts, THiGER]):
    def from_params(self, params: dict[str, Any], ckpt_path: str) -> THiGERArtifacts:
        config = TrainTHiGERConfig(**params)

        target_root = self.config.data_root_uri_template.format(
            name=config.dataset.name,
            version=config.dataset.version,
        )

        dataset = DatasetDeployment(target_root=target_root)

        logger.info(f"Loading {ckpt_path}")

        ckpt_dict = torch.load(ckpt_path, map_location=self.config.device)

        state_dict: dict[str, Any] = ckpt_dict["state_dict"]

        if self.config.ckpt_is_lightning:
            state_dict = {key[7:]: value for key, value in state_dict.items()}

        thiger = THiGER.from_config(
            config.thiger,
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            num_timestamps=dataset.num_timestamps,
            num_domains=dataset.num_domains,
        )
        thiger.load_state_dict(state_dict)
        return THiGERArtifacts(dataset=dataset, thiger=thiger)
