"""API configuration extending hakken-models ModelLoaderConfig."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from hakken_models.core.configs.model_loader import ModelLoaderConfig
from hakken_models.core.constants import ModelType
from pydantic_settings import SettingsConfigDict


def get_config() -> HakkenModelsAPIConfig:
    """Build config from environment variables and .env files.

    Loads .env first, then ENV_FILE if set (e.g. ENV_FILE=.env.segal).
    Environment variables override .env file values.
    """
    load_dotenv(override=False)
    env_file = os.getenv("ENV_FILE")
    if env_file:
        load_dotenv(env_file, override=True)
    return HakkenModelsAPIConfig()


class HakkenModelsAPIConfig(ModelLoaderConfig):
    """API configuration with model type selection.

    Inherits all loader fields from ModelLoaderConfig (device, ckpt paths,
    mlflow_run_id, run_dir, etc.) and adds model type for loader/router selection.

    Environment variables override config file values. Use ENV_FILE to load
    a specific .env file (e.g. ENV_FILE=.env.segal for SeGAL).
    """

    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: ModelType = ModelType.THIGER
