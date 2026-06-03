from __future__ import annotations

from pathlib import Path

from loguru import logger
from omegaconf import DictConfig

from kge.data_processor import KGDataProcessor
from kge.models import KGEI


def run_save_embeddings(config: DictConfig, model: KGEI, _data_processor: KGDataProcessor) -> None:
    model_ckpt_path: Path = Path(config.experiment_folder) / config.model_ckpt_path

    embeddings_folder = model_ckpt_path.parent / "embeddings" / config.model_id
    embeddings_folder.mkdir(parents=True, exist_ok=True)
    model.save_embeddings(embeddings_folder, device=config.device)

    logger.info(f"Embeddings saved to {embeddings_folder}")
