from hakken_explainer.entities.config import RunConfig
from pydantic_settings import BaseSettings


class APIConfig(BaseSettings):
    path_finder: dict
    explainer: dict
    run: RunConfig
    log_level: str = "INFO"
