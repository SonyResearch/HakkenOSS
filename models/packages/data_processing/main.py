import sys

from hydra import main
from loguru import logger
from omegaconf import OmegaConf

from data_processing.data_processor.config import DataProcessorConfig
from data_processing.factories.processor_factory import DatasetProcessorFactory

logger.remove()  # Remove default handler
logger.add(
    sys.stderr, format="<green>{time}</green> | {level} | {file}:{function}:{line} | {message}"
)


@main(config_path="configs", config_name="dummy_config_pd.yaml", version_base=None)
def run(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.info(f"Configuration File Specs:\n{OmegaConf.to_yaml(cfg)}")
    config = DataProcessorConfig(**cfg_dict)

    processor = DatasetProcessorFactory.create_processor(cfg_dict["name"], config)
    processor.process()


if __name__ == "__main__":
    run()
