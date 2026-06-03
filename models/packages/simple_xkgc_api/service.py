import os
import sys
from typing import cast

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from spaice_inference_api import Settings, create_server

from simple_xkgc_api.container import Container
from simple_xkgc_api.entities.config import APIConfig
from simple_xkgc_api.path_explainer_loader import PathExplainerLoader
from simple_xkgc_api.router import router as xkgc_router

load_dotenv()


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH", "./config"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    config: DictConfig = OmegaConf.create(cast("dict", OmegaConf.to_container(cfg, resolve=True)))
    api_config = APIConfig.model_validate(config)

    container = Container(config=api_config)
    container.init_resources()
    container.wire()
    container.wire(modules=[__name__])

    server = create_server(
        model_loader=PathExplainerLoader,
        routers=[xkgc_router],
        setup_ml_framework=None,
        wiring_config={
            "modules": [sys.modules[__name__]],
            "packages": ["simple_xkgc", "simple_xkgc_api"],
        },
        settings=Settings(SPAICE_APPLICATION_PORT=8089),
    )

    server.run()


if __name__ == "__main__":
    main()
