from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from hakken_models.models.nn import (
    Transformer,
    tx_registry,
)

TX_CONFIGS = [
    # --- cls_token aggregation ---
    {
        "name": "Transformer",
        "kwargs": {
            "embedding_dim": 64,
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.1,
            "aggregation": "cls_token",
            "use_pos_encoding": True,
        },
    },
    {
        "name": "Transformer",
        "kwargs": {
            "embedding_dim": 128,
            "num_heads": 16,
            "num_layers": 4,
            "dropout": 0.2,
            "aggregation": "cls_token",
            "use_pos_encoding": False,
        },
    },
    {
        "name": "Transformer",
        "kwargs": {
            "embedding_dim": 32,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "aggregation": "cls_token",
            "use_pos_encoding": True,
            "max_seq_len": 500,
        },
    },
    # --- attention aggregation ---
    {
        "name": "Transformer",
        "kwargs": {
            "embedding_dim": 64,
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.1,
            "use_pos_encoding": True,
            "aggregation": "attention",
        },
    },
    {
        "name": "Transformer",
        "kwargs": {
            "embedding_dim": 128,
            "num_heads": 16,
            "num_layers": 4,
            "dropout": 0.2,
            "use_pos_encoding": False,
            "aggregation": "attention",
        },
    },
    {
        "name": "Transformer",
        "kwargs": {
            "embedding_dim": 32,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "use_pos_encoding": True,
            "max_seq_len": 500,
            "aggregation": "attention",
        },
    },
    # --- mean aggregation ---
    {
        "name": "Transformer",
        "kwargs": {
            "embedding_dim": 64,
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.1,
            "use_pos_encoding": True,
            "aggregation": "mean",
        },
    },
    {
        "name": "Transformer",
        "kwargs": {
            "embedding_dim": 32,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "use_pos_encoding": False,
            "aggregation": "mean",
        },
    },
    # --- norm_first=True (Pre-LN) ---
    {
        "name": "Transformer",
        "kwargs": {
            "embedding_dim": 64,
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.1,
            "aggregation": "cls_token",
            "use_pos_encoding": True,
            "norm_first": True,
        },
    },
    {
        "name": "Transformer",
        "kwargs": {
            "embedding_dim": 64,
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.0,
            "aggregation": "attention",
            "use_pos_encoding": True,
            "norm_first": True,
        },
    },
    {
        "name": "Transformer",
        "kwargs": {
            "embedding_dim": 64,
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.0,
            "aggregation": "mean",
            "use_pos_encoding": True,
            "norm_first": True,
        },
    },
]


def _forward_temporal(module: Transformer, x: Tensor) -> Tensor:
    """x [T, E, D] -> transpose to [E, T, D], forward -> [E, D]."""
    x = x.transpose(0, 1)
    return module.forward(x)


class TestBaseTransformer:
    __test__ = True

    @pytest.fixture(params=TX_CONFIGS)
    def transformer_config(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        return cast(dict, request.param)

    @pytest.fixture()
    def transformer_model(self, transformer_config: dict) -> Transformer:
        transformer_class = tx_registry.get(transformer_config["name"])
        return transformer_class(**transformer_config["kwargs"])

    def test_forward_output_shape(
        self, transformer_config: dict, transformer_model: Transformer
    ) -> None:
        embedding_dim = transformer_config["kwargs"]["embedding_dim"]
        num_timestamps = 10
        num_entities = 5
        x = torch.randn(num_timestamps, num_entities, embedding_dim)

        output = _forward_temporal(transformer_model, x)

        assert output.shape == (num_entities, embedding_dim)
        assert isinstance(output, Tensor)

    def test_forward_different_input_sizes(
        self, transformer_config: dict, transformer_model: Transformer
    ) -> None:
        embedding_dim = transformer_config["kwargs"]["embedding_dim"]
        test_cases = [
            (1, 1, embedding_dim),
            (5, 10, embedding_dim),
            (20, 50, embedding_dim),
            (100, 1, embedding_dim),
        ]
        for num_timestamps, num_entities, _ in test_cases:
            x = torch.randn(num_timestamps, num_entities, embedding_dim)
            output = _forward_temporal(transformer_model, x)
            assert output.shape == (num_entities, embedding_dim)

    def test_gradient_flow(self, transformer_config: dict, transformer_model: Transformer) -> None:
        embedding_dim = transformer_config["kwargs"]["embedding_dim"]
        num_timestamps = 10
        num_entities = 5
        x = torch.randn(num_timestamps, num_entities, embedding_dim, requires_grad=True)

        output = _forward_temporal(transformer_model, x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        for param in transformer_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_output_no_nan_no_inf(
        self, transformer_config: dict, transformer_model: Transformer
    ) -> None:
        embedding_dim = transformer_config["kwargs"]["embedding_dim"]
        num_timestamps = 10
        num_entities = 5
        x = torch.randn(num_timestamps, num_entities, embedding_dim)
        output = _forward_temporal(transformer_model, x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_is_instance(self, transformer_model: Transformer) -> None:
        assert isinstance(transformer_model, Transformer)
        assert isinstance(transformer_model, nn.Module)

    def test_model_has_forward_method(self, transformer_model: Transformer) -> None:
        assert hasattr(transformer_model, "forward")
        assert callable(transformer_model.forward)

    def test_pos_encoding_behavior(
        self, transformer_config: dict, transformer_model: Transformer
    ) -> None:
        embedding_dim = transformer_config["kwargs"]["embedding_dim"]
        use_pos_encoding = transformer_config["kwargs"].get("use_pos_encoding", True)
        dropout = transformer_config["kwargs"].get("dropout", None)
        if dropout is not None and dropout > 0.0:
            pytest.skip("Skipping positional encoding test when dropout is enabled.")

        num_timestamps = 10
        num_entities = 5
        x = torch.randn(num_timestamps, num_entities, embedding_dim)

        output1 = _forward_temporal(transformer_model, x)
        output2 = _forward_temporal(transformer_model, x)

        assert torch.allclose(output1, output2, atol=1e-6)
        assert not torch.isnan(output1).any()
        assert not torch.isinf(output1).any()
        if use_pos_encoding:
            assert hasattr(transformer_model, "pos_encoding")
            if transformer_model.pos_encoding is not None:
                assert isinstance(transformer_model.pos_encoding, nn.Parameter)

    def test_model_specific_attributes(
        self, transformer_config: dict, transformer_model: Transformer
    ) -> None:
        aggregation = transformer_config["kwargs"].get("aggregation", "mean")
        if aggregation == "cls_token":
            assert hasattr(transformer_model, "cls_token")
            assert isinstance(transformer_model.cls_token, nn.Parameter)
        else:
            assert hasattr(transformer_model, "pool")
            if aggregation == "attention":
                assert isinstance(transformer_model.pool, nn.Module)

    def test_encoder_present(self, transformer_model: Transformer) -> None:
        assert hasattr(transformer_model, "encoder")
        assert isinstance(transformer_model.encoder, nn.TransformerEncoder)


class TestIntegration:
    def test_full_pipeline_with_registry(self) -> None:
        for config in TX_CONFIGS:
            transformer_class = tx_registry.get(config["name"])
            transformer = transformer_class(**config["kwargs"])
            embedding_dim = config["kwargs"]["embedding_dim"]
            num_timestamps = 20
            num_entities = 8
            x = torch.randn(num_timestamps, num_entities, embedding_dim)
            output = _forward_temporal(transformer, x)
            assert output.shape == (num_entities, embedding_dim)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_training_step_simulation(self) -> None:
        embedding_dim = 64
        num_timestamps = 10
        num_entities = 5

        for aggregation in ["cls_token", "attention", "mean"]:
            transformer = tx_registry.get("Transformer")(
                embedding_dim=embedding_dim,
                aggregation=aggregation,
            )
            x = torch.randn(num_timestamps, num_entities, embedding_dim, requires_grad=True)
            output = _forward_temporal(transformer, x)
            loss = output.mean()
            loss.backward()
            assert x.grad is not None
            has_grad = any(p.requires_grad and p.grad is not None for p in transformer.parameters())
            assert has_grad

    def test_seq_len_exceeds_max_raises(self) -> None:
        transformer = Transformer(embedding_dim=32, num_heads=4, max_seq_len=10)
        x = torch.randn(4, 20, 32)  # seq_len=20 > max_seq_len=10
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            transformer.forward(x)

    def test_padding_mask_cls_token(self) -> None:
        transformer = Transformer(
            embedding_dim=32, num_heads=4, num_layers=1, dropout=0.0, aggregation="cls_token"
        )
        transformer.eval()
        x = torch.randn(2, 5, 32)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[1, 3:] = True
        out = transformer.forward(x, padding_mask=mask)
        assert out.shape == (2, 32)
        assert not torch.isnan(out).any()

    def test_padding_mask_mean(self) -> None:
        transformer = Transformer(
            embedding_dim=32, num_heads=4, num_layers=1, dropout=0.0, aggregation="mean"
        )
        transformer.eval()
        x = torch.randn(2, 5, 32)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[1, 3:] = True
        out = transformer.forward(x, padding_mask=mask)
        assert out.shape == (2, 32)
        assert not torch.isnan(out).any()

    def test_padding_mask_attention(self) -> None:
        transformer = Transformer(
            embedding_dim=32, num_heads=4, num_layers=1, dropout=0.0, aggregation="attention"
        )
        transformer.eval()
        x = torch.randn(2, 5, 32)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[1, 3:] = True
        out = transformer.forward(x, padding_mask=mask)
        assert out.shape == (2, 32)
        assert not torch.isnan(out).any()

    def test_attention_pooling_fully_padded_no_nan(self) -> None:
        """Fully-padded sequence should produce zeros, not NaN."""
        transformer = Transformer(
            embedding_dim=32, num_heads=4, num_layers=1, dropout=0.0, aggregation="attention"
        )
        transformer.eval()
        x = torch.randn(1, 4, 32)
        mask = torch.ones(1, 4, dtype=torch.bool)  # everything padded
        out = transformer.forward(x, padding_mask=mask)
        assert not torch.isnan(out).any()

    def test_norm_first_has_final_layernorm(self) -> None:
        transformer = Transformer(embedding_dim=64, num_heads=8, num_layers=2, norm_first=True)
        assert transformer.encoder.norm is not None
        assert isinstance(transformer.encoder.norm, nn.LayerNorm)

    def test_norm_first_false_no_extra_layernorm(self) -> None:
        transformer = Transformer(embedding_dim=64, num_heads=8, num_layers=2, norm_first=False)
        assert transformer.encoder.norm is None
