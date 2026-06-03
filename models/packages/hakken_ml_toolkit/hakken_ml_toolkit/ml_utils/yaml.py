from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from hakken_ml_toolkit.ml_utils.exceptions import InvalidDictConfigError, InvalidYamlError

if TYPE_CHECKING:
    from pathlib import Path


class YAMLUtils:
    @staticmethod
    def load(file_path: str | Path) -> dict[str, Any]:
        with open(file_path) as f:
            my_dict: dict[str, Any] = yaml.safe_load(f)
            return my_dict

    @staticmethod
    def hydra_load(file_path: Path) -> DictConfig:
        folder = str(file_path.parent)
        file_name = file_path.stem
        with hydra.initialize(version_base=None, config_path=str(folder)):
            cfg = hydra.compose(config_name=file_name)

        config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

        return cast("DictConfig", config)

    @staticmethod
    def omegaconf_load(file_path: Path) -> DictConfig:
        cfg_unresolved = OmegaConf.load(str(file_path))

        cfg = OmegaConf.create(OmegaConf.to_container(cfg_unresolved, resolve=True))

        assert isinstance(cfg, DictConfig)
        return cfg

    @staticmethod
    def load_many(yaml_list: list[str]) -> DictConfig:
        cfg = None
        for yaml_file in yaml_list:
            try:
                if cfg is None:
                    cfg = OmegaConf.load(yaml_file)
                else:
                    cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(yaml_file))
            except Exception as err:
                raise InvalidYamlError() from err

        if cfg is None:
            raise InvalidYamlError()
        if not isinstance(cfg, DictConfig):
            raise InvalidDictConfigError()
        return cfg
