from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from kge.common.entities import (
    KGEPredictRequest,
    KGEPredictResponse,
    KGEScoreIndexRequest,
    KGEScoreResponse,
)
from kge.data_processor import KGDataProcessor
from kge.models.base import KGEI
from spaice_inference_api import ILogger
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from kge_api.config import APIConfig
from kge_api.kge_loader import KGEExperimentData
from kge_api.router import device, predict, score_from_index


class TestRouterEndpoints:
    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        return MagicMock(spec=ILogger)

    @pytest.fixture
    def mock_experiment_data(self) -> MagicMock:
        mock_data = MagicMock(spec=KGEExperimentData)
        # Setup model with parameters that have a device attribute
        model_param = MagicMock()
        model_param.device = "cpu"
        mock_data.model = MagicMock(spec=KGEI)
        mock_data.model.parameters.return_value = [model_param]
        mock_data.data_processor = MagicMock(spec=KGDataProcessor)

        return mock_data

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> APIConfig:
        return APIConfig(experiment_folder=tmp_path, device="cpu")

    @patch("kge_api.router.KGEInferenceActions")
    def test_predict_endpoint_success(
        self,
        mock_predict_action: MagicMock,
        mock_logger: MagicMock,
        mock_experiment_data: MagicMock,
        mock_config: APIConfig,
    ) -> None:
        # Setup
        request = KGEPredictRequest(
            subject_id_list=["subject1", "subject2"],
            object_id_list=["object1", "object2"],
            relation_id_list=["relation1", "relation2"],
            inference_config=None,
        )
        expected_response = KGEPredictResponse(
            relations_ids=["relation1", "relation2"],
            relations_probs=[[0.8, 0.2], [0.6, 0.4]],
            relations_scores=[[0.9, 0.1], [0.7, 0.3]],
        )
        mock_predict_action.predict.return_value = expected_response

        # Act
        with patch("kge_api.router.Provide", MagicMock(return_value=mock_config)):
            result = predict(request, mock_logger, mock_experiment_data, mock_config)

        # Assert
        mock_logger.debug.assert_called_once()
        mock_predict_action.predict.assert_called_once_with(
            request=request,
            kge=mock_experiment_data.model,
            data_processing=mock_experiment_data.data_processor,
            device=mock_config.device,
        )
        assert result == expected_response
        assert result.relations_ids == ["relation1", "relation2"]
        assert result.relations_probs == [[0.8, 0.2], [0.6, 0.4]]
        assert result.relations_scores == [[0.9, 0.1], [0.7, 0.3]]

    @patch("kge_api.router.KGEInferenceActions")
    def test_predict_endpoint_error(
        self,
        mock_predict_action: MagicMock,
        mock_logger: MagicMock,
        mock_experiment_data: MagicMock,
        mock_config: APIConfig,
    ) -> None:
        # Setup
        request = KGEPredictRequest(
            subject_id_list=["subject1"],
            object_id_list=["object1"],
            relation_id_list=None,
            inference_config=None,
        )
        mock_predict_action.predict.side_effect = ValueError("Test error")

        # Act & Assert
        with patch("kge_api.router.Provide", MagicMock(return_value=mock_config)):
            with pytest.raises(HTTPException) as exc_info:
                predict(request, mock_logger, mock_experiment_data, mock_config)

            assert exc_info.value.status_code == HTTP_500_INTERNAL_SERVER_ERROR
            assert "Test error" in str(exc_info.value.detail)

    @patch("kge_api.router.KGEInferenceActions")
    def test_score_from_index_endpoint(
        self,
        mock_predict_action: MagicMock,
        mock_logger: MagicMock,
        mock_experiment_data: MagicMock,
        mock_config: APIConfig,
    ) -> None:
        # Setup
        request = KGEScoreIndexRequest(facts_index_list=[[1, 2, 3], [4, 5, 6]])
        expected_response = KGEScoreResponse(scores_list=[0.85, 0.72], normalized_scores_list=None)
        mock_predict_action.score_from_index.return_value = expected_response

        # Act
        with patch("kge_api.router.Provide", MagicMock(return_value=mock_config)):
            result = score_from_index(request, mock_logger, mock_experiment_data, mock_config)

        # Assert
        mock_logger.debug.assert_called_once()
        mock_predict_action.score_from_index.assert_called_once_with(
            request=request, kge=mock_experiment_data.model, device=mock_config.device
        )
        assert result == expected_response
        assert result.scores_list == [0.85, 0.72]

    def test_device_endpoint(self, mock_logger: MagicMock, mock_experiment_data: MagicMock) -> None:
        # Act
        result = device(mock_logger, mock_experiment_data)

        # Assert
        assert result == {"cpu"}
        mock_experiment_data.model.parameters.assert_called_once()
