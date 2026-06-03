from __future__ import annotations

import ast
import os
import shutil

from loguru import logger
from omegaconf import OmegaConf


class IOUtils:
    @staticmethod
    def print_omegaconf(cfg: OmegaConf) -> None:
        yaml_string = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)

        logger.info(yaml_string)

    @staticmethod
    def makedirs(file_path: str, exist_ok: bool = True) -> None:
        os.makedirs(file_path, exist_ok=exist_ok)

    @staticmethod
    def txt_to_list(file_path: str, literal: bool = True) -> list[str]:
        with open(file_path) as f:
            lines = f.readlines()

        output = []
        for x_raw in lines:
            x = x_raw
            if literal:
                x = x.replace("nan", '"nan"')
                output.append(ast.literal_eval(x.strip()))
            else:
                output.append(x.strip())
        return output

    @staticmethod
    def rm_if_exists(folder: str):
        if os.path.exists(folder):
            shutil.rmtree(folder)
