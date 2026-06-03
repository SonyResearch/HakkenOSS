import os
import sys

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from spaice_inference_api import Settings, create_server

from kge_api.config import APIConfig
from kge_api.container import Container
from kge_api.kge_loader import KGERunLoader
from kge_api.router import router as kge_router

load_dotenv(override=False)


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH", "./config"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    api_config = APIConfig.model_validate(cfg)
    container = Container(config=api_config)
    container.init_resources()
    container.wire()
    container.wire(modules=[__name__])

    server = create_server(
        model_loader=KGERunLoader,
        routers=[kge_router],
        setup_ml_framework=None,
        wiring_config={
            "modules": [sys.modules[__name__]],
            "packages": ["kge_api", "datasets"],
        },
        settings=Settings(),
    )

    server.run()


if __name__ == "__main__":
    main()
