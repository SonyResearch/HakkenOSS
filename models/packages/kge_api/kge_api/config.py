from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings


class APIConfig(BaseSettings):
    experiment_folder: Path
    config_path: str = ".hydra/config.yaml"
    model_ckpt_path: str = "seed_0/model_checkpoint/last.ckpt"
    score_scaler_json_path: str | None = None
    model_ckpt_is_lightning: bool = True
    device: Literal["cpu", "cuda"] = "cuda"

    @field_validator("experiment_folder")
    @classmethod
    def validate_paths(cls, v: Path) -> Path:
        if not v.exists():
            msg = f"Path {v} does not exist"
            raise ValueError(msg)
        return v
