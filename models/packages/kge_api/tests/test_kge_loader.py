from pathlib import Path
from unittest.mock import MagicMock, patch

from kge.common.actions.kge_loader_action import KGEExperimentData
from spaice_inference_api import ILogger, ModelLoadingOptions

from kge_api.config import APIConfig
from kge_api.kge_loader import KGERunLoader


def test_load_experiment(tmp_path: Path) -> None:
    # Setup

    # Arrange
    loader = KGERunLoader()
    options = MagicMock(spec=ModelLoadingOptions)
    logger = MagicMock(spec=ILogger)

    tmp_path = tmp_path / "experiment"
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = APIConfig(experiment_folder=tmp_path)
    # Act
    with patch(
        "kge_api.kge_loader.KGELoader.load_experiment",
        return_value=KGEExperimentData(model=MagicMock(), data_processor=MagicMock()),
    ) as mock_load_experiment:
        # Call the load method
        _ = loader.load(options, logger, config=config)
        # Assert
        mock_load_experiment.assert_called_once_with(
            experiment_folder=config.experiment_folder,
            config_path=config.config_path,
            model_ckpt_path=config.model_ckpt_path,
            model_ckpt_is_lightning=config.model_ckpt_is_lightning,
            device=config.device,
            load_negative_sampler=True,
        )
