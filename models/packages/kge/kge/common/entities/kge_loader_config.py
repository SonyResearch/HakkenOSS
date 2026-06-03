from pathlib import Path

import torch
from pydantic import BaseModel, Field


class KGELoadExperimentConfig(BaseModel):
    experiment_folder: str | Path = Field(
        ..., description="Root folder of the experiment (must exist)."
    )
    config_path: str = Field(
        default=".hydra/config.yaml",
        description="Relative path to the config file within the experiment folder.",
    )
    model_ckpt_path: str = Field(
        default="seed_0/model_checkpoint/last.ckpt",
        description="Relative path to the model checkpoint file within the experiment folder.",
    )
    model_ckpt_is_lightning: bool = Field(
        default=True,
        description="Whether the checkpoint is a PyTorch Lightning checkpoint.",
    )
    device: str | torch.device = Field(
        default="cpu",
        description="Device to load the model on (e.g., 'cpu', 'cuda', 'cuda:0').",
    )

    class Config:
        arbitrary_types_allowed = True  # Allows torch.device type
