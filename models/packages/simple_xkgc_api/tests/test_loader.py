from __future__ import annotations

from unittest.mock import MagicMock, patch

from simple_xkgc_api.entities.config import APIConfig, RunConfig
from simple_xkgc_api.path_explainer_loader import PathExplainerLoader

EXPECTED_NUM_CALLS = 2


def test_load_returns_configured_explainer():
    # Arrange
    loader = PathExplainerLoader()

    logger = MagicMock()
    logger.info = MagicMock()

    config = APIConfig(
        path_finder={"_target_": "simple_xkgc.PathFinderImpl"},
        explainer={"_target_": "simple_xkgc.PathExplainerImpl"},
        run=RunConfig(device="cpu", batch_size=1),
    )

    # Mocks returned by hydra.utils.instantiate
    mock_path_finder = MagicMock(name="PathFinder")
    mock_explainer = MagicMock(name="PathExplainer")

    # Make sure explainer.setup() is observable
    mock_explainer.setup = MagicMock()

    # hydra.utils.instantiate should return:
    #   first call -> path_finder
    #   second call -> explainer
    with patch("simple_xkgc_api.path_explainer_loader.hydra.utils.instantiate") as inst:
        inst.side_effect = [mock_path_finder, mock_explainer]

        # Act
        # _options is not used in the method, so any object is fine
        result = loader.load(_options=object(), logger=logger, config=config)

    # Assert
    # Returned object is the explainer instance
    assert result is mock_explainer

    # hydra instantiate called twice with the two config nodes
    assert inst.call_count == EXPECTED_NUM_CALLS
    inst.assert_any_call(config.path_finder)
    inst.assert_any_call(config.explainer)

    # explainer.setup must be called with the path_finder instance
    mock_explainer.setup.assert_called_once_with(path_finder=mock_path_finder)

    # We at least logged the config once
    logger.info.assert_called()
