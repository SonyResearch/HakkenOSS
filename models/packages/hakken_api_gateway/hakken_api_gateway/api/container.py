from dependency_injector import containers, providers
from pydantic_settings import BaseSettings, SettingsConfigDict

from hakken_api_gateway.core.entities.config import DatabaseConfig


class ApiConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_config: DatabaseConfig


class DataApiContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=["hakken_api_gateway"])
    # The following is so that we can pass the container itself
    # as a dependency
    __self__: providers.Self["DataApiContainer"] = providers.Self()

    config = providers.Dependency(ApiConfig)
