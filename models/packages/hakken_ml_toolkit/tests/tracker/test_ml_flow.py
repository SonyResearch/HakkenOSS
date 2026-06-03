from unittest import mock

import pandas as pd
import pytest
from mlflow.tracking.fluent import ActiveRun

from hakken_ml_toolkit.tracker.impl.ml_flow import MLFlowTracker, MLFlowTrackerConfig


@pytest.fixture
def mock_mlflow():
    with mock.patch("hakken_ml_toolkit.tracker.impl.ml_flow.mlflow") as mock_mlflow:
        mock_active_run = mock.MagicMock(spec=ActiveRun)
        mock_mlflow.start_run.return_value = mock_active_run

        yield mock_mlflow


@pytest.fixture
def mock_random_utils():
    with mock.patch("hakken_ml_toolkit.tracker.impl.ml_flow.RandomUtils") as mock_random_utils:
        mock_random_utils.generate_run_name.return_value = "test-run-123"
        yield mock_random_utils


class TestMLFlowTracker:
    def test_init(self, mock_mlflow, mock_random_utils):
        config = MLFlowTrackerConfig(
            experiment_name="test-experiment",
            run_name="test-run",
            tracking_uri="http://test-mlflow-server:5000",
        )

        tracker = MLFlowTracker(config)

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://test-mlflow-server:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")
        mock_random_utils.generate_run_name.assert_called_once_with(
            basename="test-run", use_local_random=True
        )
        assert tracker.run_name == "test-run-123"
        assert tracker.run is None

    def test_init_without_tracking_uri(self, mock_mlflow, mock_random_utils):
        config = MLFlowTrackerConfig(experiment_name="test-experiment", run_name="test-run")

        tracker = MLFlowTracker(config)
        assert tracker is not None

        mock_mlflow.set_tracking_uri.assert_not_called()
        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")
        mock_random_utils.generate_run_name.assert_called_once_with(
            basename="test-run", use_local_random=True
        )

    # ruff: noqa: ARG002
    def test_context_manager(self, mock_mlflow, mock_random_utils):
        config = MLFlowTrackerConfig(experiment_name="test-experiment", run_name="test-run")

        with MLFlowTracker(config) as tracker:
            mock_mlflow.start_run.assert_called_once_with(run_name="test-run-123")
            assert tracker.run is mock_mlflow.start_run.return_value

        mock_mlflow.end_run.assert_called_once()

    def test_track_value(self, mock_mlflow):
        config = MLFlowTrackerConfig(
            experiment_name="test-experiment", run_name="test-run", persist=True
        )

        with MLFlowTracker(config) as tracker:
            tracker.step = 5
            tracker._track_value("accuracy", 0.95)
            mock_mlflow.log_metric.assert_called_once_with("accuracy", 0.95, step=5)

            mock_mlflow.log_metric.reset_mock()
            tracker._track_value("loss", 0.1, step=10)
            mock_mlflow.log_metric.assert_called_once_with("loss", 0.1, step=10)

    def test_track_value_no_persist(self, mock_mlflow):
        config = MLFlowTrackerConfig(
            experiment_name="test-experiment", run_name="test-run", persist=False
        )

        with MLFlowTracker(config) as tracker:
            tracker._track_value("accuracy", 0.95)
            mock_mlflow.log_metric.assert_not_called()

    def test_track_data(self, mock_mlflow):
        config = MLFlowTrackerConfig(
            experiment_name="test-experiment", run_name="test-run", persist=True
        )

        with MLFlowTracker(config) as tracker:
            tracker.step = 5
            tracker._track_data({"accuracy": 0.95, "loss": 0.1})

            mock_mlflow.log_metric.assert_any_call("accuracy", 0.95, step=5)
            mock_mlflow.log_metric.assert_any_call("loss", 0.1, step=5)
            assert mock_mlflow.log_metric.call_count == 2

            # Test with explicit step
            mock_mlflow.log_metric.reset_mock()
            tracker._track_data({"accuracy": 0.97, "loss": 0.05}, step=10)
            mock_mlflow.log_metric.assert_any_call("accuracy", 0.97, step=10)
            mock_mlflow.log_metric.assert_any_call("loss", 0.05, step=10)

    def test_track_config(self, mock_mlflow):
        config = MLFlowTrackerConfig(
            experiment_name="test-experiment", run_name="test-run", persist=True
        )

        with MLFlowTracker(config) as tracker:
            tracker._track_config({"learning_rate": 0.01, "batch_size": 32})
            mock_mlflow.log_params.assert_called_once_with(
                {"learning_rate": 0.01, "batch_size": 32}
            )

    def test_track_table(self, mock_mlflow):
        config = MLFlowTrackerConfig(
            experiment_name="test-experiment", run_name="test-run", persist=True
        )

        with MLFlowTracker(config) as tracker:
            columns = ["x", "y"]
            data = [[1, 2], [3, 4]]
            tracker.track_table("coordinates", columns, data)

            # Verify pandas DataFrame was created correctly and logged
            mock_mlflow.log_table.assert_called_once()
            call_args = mock_mlflow.log_table.call_args[1]

            # Check artifact file name
            assert call_args["artifact_file"] == "coordinates.json"

            # Check DataFrame content
            pd.testing.assert_frame_equal(
                call_args["data"], pd.DataFrame(columns=columns, data=data)
            )

            # Test with step
            mock_mlflow.log_table.reset_mock()
            tracker.track_table("coordinates", columns, data, step=5)

            call_args = mock_mlflow.log_table.call_args[1]
            assert call_args["artifact_file"] == "coordinates__5.json"

    def test_finish(self, mock_mlflow):
        config = MLFlowTrackerConfig(experiment_name="test-experiment", run_name="test-run")

        tracker = MLFlowTracker(config)
        tracker.run = mock.MagicMock()

        tracker.finish()
        mock_mlflow.end_run.assert_called_once()
        assert tracker.run is None

        mock_mlflow.end_run.reset_mock()
        tracker.finish()
        mock_mlflow.end_run.assert_not_called()

    def test_del(self, mock_mlflow):
        config = MLFlowTrackerConfig(experiment_name="test-experiment", run_name="test-run")

        tracker = MLFlowTracker(config)
        tracker.run = mock.MagicMock()

        tracker.__del__()
        mock_mlflow.end_run.assert_called_once()
        assert tracker.run is None
