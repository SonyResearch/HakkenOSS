from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Class representing application settings"""

    HOST: str = "0.0.0.0"
    SPAICE_APPLICATION_PORT: int = 8088
    SPAICE_MODEL_NAME: str = "test"
    SPAICE_MODEL_PATH: str = "dummy path to model"
    SPAICE_INFERENCE_API_KEY: str | None = None
    UVICORN_KEEP_ALIVE_TIMEOUT: int = 5
    UVICORN_CONCURRENCY: int | None = None
    UVICORN_CAPACITY_LIMITER: int = 5

    # Use memory to set a threshold for gpu memory allocation when using PyTorch
    # Use fraction to set a threshold for gpu memory allocation when using TF
    GPU_MEMORY: int | None = None
    GPU_FRACTION: float | None = None


SettingsToken = "settings"
