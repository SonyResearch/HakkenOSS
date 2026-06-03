#!/usr/bin/env python3
"""Run the Element Resolver API server with Hydra config."""

import os
from pathlib import Path

import hydra
import uvicorn
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from hakken_agents.api.app import create_app
from hakken_agents.api.config import ElementResolverAPIConfig

load_dotenv(override=False)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = os.getenv("CONFIG_PATH") or str(_PROJECT_ROOT / "configs")


@hydra.main(
    version_base=None,
    config_path=_CONFIG_PATH,
    config_name="element_resolver_api",
)
def main(cfg: DictConfig) -> None:
    """Load config from Hydra YAML and start the API server."""
    # Resolve OmegaConf interpolations (e.g. ${oc.env:VAR})
    OmegaConf.resolve(cfg)
    config = ElementResolverAPIConfig.model_validate(OmegaConf.to_container(cfg))
    logger.info(config.model_dump_json(indent=2))
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
