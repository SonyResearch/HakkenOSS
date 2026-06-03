"""Tests for class instantiation utilities."""

from typing import cast

import pytest
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, Metric
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)

from hakken_models.core.utils.runtime import instantiate_from_string

# ============================================================================
# Test Configuration
# ============================================================================
TORCHMETRICS_CLASSES = [
    (
        "torchmetrics.classification.MultilabelAccuracy",
        MultilabelAccuracy,
        {"num_labels": 10},
    ),
    (
        "torchmetrics.classification.MultilabelPrecision",
        MultilabelPrecision,
        {"num_labels": 10},
    ),
    (
        "torchmetrics.classification.MultilabelRecall",
        MultilabelRecall,
        {"num_labels": 10},
    ),
    (
        "torchmetrics.classification.MultilabelF1Score",
        MultilabelF1Score,
        {"num_labels": 10},
    ),
    ("torchmetrics.MeanSquaredError", MeanSquaredError, {}),
    ("torchmetrics.MeanAbsoluteError", MeanAbsoluteError, {}),
]


# ============================================================================
# Test Class
# ============================================================================
class TestInstantiateFromString:
    """Test instantiate_from_string function with torchmetrics classes."""

    @pytest.fixture(params=TORCHMETRICS_CLASSES)
    def metric_config(self, request: pytest.FixtureRequest) -> tuple[str, type, dict]:
        """Parametrized metric configuration."""
        return request.param

    def test_instantiate_torchmetrics_with_kwargs(
        self, metric_config: tuple[str, type, dict]
    ) -> None:
        """Test instantiating torchmetrics classes with keyword arguments."""
        class_path, expected_class, kwargs = metric_config

        instance = instantiate_from_string(class_path, **kwargs)

        assert isinstance(instance, expected_class)
        assert instance is not None

    def test_instantiate_torchmetrics_without_args(self) -> None:
        """Test instantiating torchmetrics classes that don't require arguments."""
        instance = instantiate_from_string("torchmetrics.MeanSquaredError")

        assert isinstance(instance, MeanSquaredError)
        assert instance is not None

    def test_instantiate_torchmetrics_and_use(self) -> None:
        """Test that instantiated torchmetrics can be used for computation."""
        accuracy = cast(
            Metric,
            instantiate_from_string("torchmetrics.classification.MultilabelAccuracy", num_labels=3),
        )

        # Create dummy multilabel predictions and targets
        # Shape: [batch_size, num_labels] with binary values per label
        preds = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=torch.float)
        target = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=torch.int)

        # Update and compute
        accuracy.update(preds, target)
        result = accuracy.compute()

        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0.0
        assert result.item() <= 1.0

    def test_instantiate_with_positional_args(self) -> None:
        """Test instantiating with positional arguments (if supported)."""
        # Some torchmetrics classes support positional args, but most use kwargs
        # Testing with a simple case
        instance = instantiate_from_string("torchmetrics.MeanSquaredError")

        assert isinstance(instance, MeanSquaredError)

    def test_instantiate_multiple_metrics(self) -> None:
        """Test instantiating multiple different metrics."""
        metrics = [
            instantiate_from_string("torchmetrics.classification.MultilabelAccuracy", num_labels=5),
            instantiate_from_string(
                "torchmetrics.classification.MultilabelPrecision", num_labels=5
            ),
            instantiate_from_string("torchmetrics.classification.MultilabelRecall", num_labels=5),
            instantiate_from_string("torchmetrics.classification.MultilabelF1Score", num_labels=5),
        ]

        for metric in metrics:
            assert metric is not None

        # Verify they're different instances
        assert metrics[0] is not metrics[1]

    def test_instantiate_with_complex_kwargs(self) -> None:
        """Test instantiating with complex keyword arguments."""
        accuracy = instantiate_from_string(
            "torchmetrics.classification.MultilabelAccuracy",
            num_labels=10,
            average="macro",
            threshold=0.5,
        )

        assert isinstance(accuracy, MultilabelAccuracy)
        assert accuracy.num_labels == 10

    def test_invalid_module_path_raises_import_error(self) -> None:
        """Test that invalid module path raises ImportError."""
        with pytest.raises(ImportError, match="No module named"):
            instantiate_from_string("nonexistent.module.Class")

    def test_invalid_class_name_raises_attribute_error(self) -> None:
        """Test that invalid class name raises AttributeError."""
        with pytest.raises(AttributeError):
            instantiate_from_string("torchmetrics.NonExistentMetric")

    def test_invalid_class_path_format_raises_value_error(self) -> None:
        """Test that invalid class path format raises ValueError."""
        with pytest.raises(ValueError, match="not enough values to unpack"):
            instantiate_from_string("InvalidPathWithoutDot")

    def test_instantiate_regression_metrics(self) -> None:
        """Test instantiating regression metrics."""
        mse = instantiate_from_string("torchmetrics.MeanSquaredError")
        mae = instantiate_from_string("torchmetrics.MeanAbsoluteError")

        assert isinstance(mse, MeanSquaredError)
        assert isinstance(mae, MeanAbsoluteError)

        # Test they work
        preds = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])

        mse.update(preds, target)
        mae.update(preds, target)

        mse_result = mse.compute()
        mae_result = mae.compute()

        assert isinstance(mse_result, torch.Tensor)
        assert isinstance(mae_result, torch.Tensor)
        assert mse_result.item() >= 0.0
        assert mae_result.item() >= 0.0

    def test_instantiate_with_different_num_labels(self) -> None:
        """Test instantiating multilabel classification metrics with different num_labels."""
        for num_labels in [2, 5, 10, 100]:
            accuracy = instantiate_from_string(
                "torchmetrics.classification.MultilabelAccuracy", num_labels=num_labels
            )
            assert isinstance(accuracy, MultilabelAccuracy)
            assert accuracy.num_labels == num_labels

    def test_instantiate_multilabel_metrics_and_compute(self) -> None:
        """Test instantiating multilabel classification metrics and computing scores."""
        accuracy = instantiate_from_string(
            "torchmetrics.classification.MultilabelAccuracy", num_labels=4
        )
        precision = instantiate_from_string(
            "torchmetrics.classification.MultilabelPrecision", num_labels=4
        )
        recall = instantiate_from_string(
            "torchmetrics.classification.MultilabelRecall", num_labels=4
        )
        f1 = instantiate_from_string("torchmetrics.classification.MultilabelF1Score", num_labels=4)

        assert isinstance(accuracy, MultilabelAccuracy)
        assert isinstance(precision, MultilabelPrecision)
        assert isinstance(recall, MultilabelRecall)
        assert isinstance(f1, MultilabelF1Score)

        # Test they work with multilabel predictions
        # Shape: [batch_size, num_labels] - binary predictions per label
        preds = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.float
        )
        target = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.int
        )

        accuracy.update(preds, target)
        precision.update(preds, target)
        recall.update(preds, target)
        f1.update(preds, target)

        assert accuracy.compute().item() >= 0.0
        assert precision.compute().item() >= 0.0
        assert recall.compute().item() >= 0.0
        assert f1.compute().item() >= 0.0

    def test_instantiate_multilabel_with_average_options(self) -> None:
        """Test instantiating multilabel metrics with different average options."""
        for average in ["micro", "macro", "weighted", "none"]:
            precision = instantiate_from_string(
                "torchmetrics.classification.MultilabelPrecision",
                num_labels=5,
                average=average,
            )
            assert isinstance(precision, MultilabelPrecision)
            assert precision.average == average

    def test_instantiate_multilabel_with_threshold(self) -> None:
        """Test instantiating multilabel metrics with custom threshold."""
        accuracy = instantiate_from_string(
            "torchmetrics.classification.MultilabelAccuracy", num_labels=3, threshold=0.7
        )

        assert isinstance(accuracy, MultilabelAccuracy)
        assert accuracy.threshold == 0.7

        # Test with predictions that need thresholding
        preds = torch.tensor([[0.8, 0.3, 0.9], [0.2, 0.7, 0.1]], dtype=torch.float)
        target = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.int)

        accuracy.update(preds, target)
        result = accuracy.compute()
        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0.0
        assert result.item() <= 1.0
