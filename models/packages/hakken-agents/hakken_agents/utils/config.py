"""
Hydra configuration composition utilities.

This module provides helper functions for composing Hydra configs
while integrating with Typer CLI commands.
"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


def compose_config(
    config_name: str = "defaults",
    config_dir: Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Compose config using Hydra (handles defaults automatically)."""

    if config_dir is None:
        config_dir = Path("configs")

    # Ensure config_dir is absolute
    config_dir = config_dir.resolve()

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Initialize Hydra with the config directory
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # Compose config with overrides
        overrides_list = overrides or []
        cfg = compose(config_name=config_name, overrides=overrides_list)

        # Resolve interpolations
        OmegaConf.resolve(cfg)

    return cfg


def config_to_dict(cfg: DictConfig) -> dict:
    """Convert OmegaConf DictConfig to a plain Python dictionary."""
    return OmegaConf.to_container(cfg, resolve=True)
