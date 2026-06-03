"""Generic pytest test suite for KGEI implementations.

This module provides a comprehensive test suite that can be used to test
any implementation of the KGEI interface. To use it, create a subclass
and provide a model fixture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch.utils.data import TensorDataset

from kge.common.types import LongTensor2D
from kge.models.base import KGEI
from kge.models.config import KGEConfig

if TYPE_CHECKING:
    from pathlib import Path


class BaseKGEITest:
    """Base test class for KGEI implementations.

    Subclass this and provide a `model` fixture to test any KGEI implementation.

    Example:
        class TestRandomKGE(BaseKGEITest):
            @pytest.fixture
            def model(self):
                from kge.models.random_kge import RandomKGE
                config = KGEConfig(num_entities=100, num_relations=10, embedding_dim=32)
                return RandomKGE(config)
    """

    @pytest.fixture
    def model(self) -> KGEI[KGEConfig]:
        """Fixture to be overridden by subclasses to provide specific model instances."""
        raise NotImplementedError(
            "Subclasses must provide a model fixture that returns a KGEI instance."
        )

    @pytest.fixture
    def batch_size(self) -> int:
        """Default batch size for tests."""
        return 16

    @pytest.fixture
    def sro_batch(self, model: KGEI[KGEConfig], batch_size: int) -> LongTensor2D:
        """Create a batch of subject-relation-object triples."""
        return torch.randint(
            0,
            min(model.config.num_entities, model.config.num_relations),
            (batch_size, 3),
            dtype=torch.long,
        )

    @pytest.fixture
    def sr_batch(self, model: KGEI[KGEConfig], batch_size: int) -> LongTensor2D:
        """Create a batch of subject-relation pairs."""
        return torch.randint(
            0,
            min(model.config.num_entities, model.config.num_relations),
            (batch_size, 2),
            dtype=torch.long,
        )

    @pytest.fixture
    def ro_batch(self, model: KGEI[KGEConfig], batch_size: int) -> LongTensor2D:
        """Create a batch of relation-object pairs."""
        return torch.randint(
            0,
            min(model.config.num_entities, model.config.num_relations),
            (batch_size, 2),
            dtype=torch.long,
        )

    @pytest.fixture
    def so_batch(self, model: KGEI[KGEConfig], batch_size: int) -> LongTensor2D:
        """Create a batch of subject-object pairs."""
        return torch.randint(
            0,
            model.config.num_entities,
            (batch_size, 2),
            dtype=torch.long,
        )

    # ========== Basic Functionality Tests ==========

    def test_initialization(self, model: KGEI[KGEConfig]):
        """Test that model initializes correctly with config."""
        assert model.config is not None
        assert hasattr(model.config, "num_entities")
        assert hasattr(model.config, "num_relations")
        assert hasattr(model.config, "embedding_dim")
        assert model.config.num_entities > 0
        assert model.config.num_relations > 0
        assert model.config.embedding_dim > 0

    def test_embedding_dim(self, model: KGEI[KGEConfig]):
        """Test embedding_dim property."""
        assert model.embedding_dim() == model.config.embedding_dim

    def test_config_class(self, model: KGEI[KGEConfig]):
        """Test that get_config_class returns the correct type."""
        config_class = model.get_config_class()
        assert issubclass(config_class, KGEConfig)
        assert isinstance(model.config, config_class)

    # ========== Device Tests ==========

    def test_device_property(self, model: KGEI[KGEConfig]):
        """Test that device property returns a valid device."""
        device = model.device
        assert isinstance(device, str | torch.device)

        # Should be on CPU by default
        assert device in ("cpu", torch.device("cpu"))

    def test_to_device_cpu(self, model: KGEI[KGEConfig]):
        """Test moving model to CPU device."""
        model_cpu = model.to_device("cpu")
        assert model_cpu is model  # Should return self
        assert model_cpu.device in ("cpu", torch.device("cpu"))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_cuda(self, model: KGEI[KGEConfig]):
        """Test moving model to CUDA device if available."""
        model_cuda = model.to_device("cuda")
        assert model_cuda is model
        assert model_cuda.device in ("cuda:0", torch.device("cuda:0")), (
            f"device: {model_cuda.device}"
        )

    # ========== Model State Tests ==========

    def test_eval_mode(self, model: KGEI[KGEConfig]):
        """Test setting model to eval mode."""
        model_eval = model.eval()
        assert model_eval is model  # Should return self
        assert not model_eval.training

    def test_train_mode(self, model: KGEI[KGEConfig]):
        """Test setting model to train mode."""
        model_train = model.train()
        assert model_train is model
        assert model_train.training

    def test_train_mode_toggle(self, model: KGEI[KGEConfig]):
        """Test toggling between train and eval modes."""
        model.eval()
        assert not model.training
        model.train(True)
        assert model.training
        model.train(False)
        assert not model.training

    # ========== Forward Pass Tests ==========

    def test_forward(self, model: KGEI[KGEConfig], sro_batch: LongTensor2D):
        """Test forward pass returns valid output."""
        model.eval()
        output = model.forward(sro_batch)
        assert output is not None
        # KGEForwardOutput should have a scores attribute
        assert hasattr(output, "scores")
        assert isinstance(output.scores, torch.Tensor)
        assert output.scores.shape[0] == sro_batch.shape[0]

    def test_call_method(self, model: KGEI[KGEConfig], sro_batch: LongTensor2D):
        """Test that __call__ method works (should call forward)."""
        model.eval()
        output = model(sro_batch)
        assert output is not None
        assert hasattr(output, "scores")

    # ========== Embedding Tests ==========

    def test_entity_embeddings(self, model: KGEI[KGEConfig]):
        """Test entity embeddings method."""
        batch_size = 8
        entity_ids = torch.randint(0, model.config.num_entities, (batch_size,), dtype=torch.long)

        embeddings = model.entity_embeddings(entity_ids)
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (batch_size, model.config.embedding_dim)
        assert embeddings.dtype == torch.float32

    def test_relation_embeddings(self, model: KGEI[KGEConfig]):
        """Test relation embeddings method."""
        batch_size = 8
        relation_ids = torch.randint(0, model.config.num_relations, (batch_size,), dtype=torch.long)

        embeddings = model.relation_embeddings(relation_ids)
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (batch_size, model.config.embedding_dim)
        assert embeddings.dtype == torch.float32

    def test_entity_embeddings_all_entities(self, model: KGEI[KGEConfig]):
        """Test entity embeddings for all entities."""
        all_entities = torch.arange(model.config.num_entities, dtype=torch.long)
        embeddings = model.entity_embeddings(all_entities)
        assert embeddings.shape == (model.config.num_entities, model.config.embedding_dim)

    def test_relation_embeddings_all_relations(self, model: KGEI[KGEConfig]):
        """Test relation embeddings for all relations."""
        all_relations = torch.arange(model.config.num_relations, dtype=torch.long)
        embeddings = model.relation_embeddings(all_relations)
        assert embeddings.shape == (model.config.num_relations, model.config.embedding_dim)

    # ========== Scoring Tests ==========

    def test_score(self, model: KGEI[KGEConfig], sro_batch: LongTensor2D):
        """Test score method for full triples."""
        model.eval()
        scores = model.score(sro_batch)
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (sro_batch.shape[0], 1)
        assert scores.dtype == torch.float32

    def test_score_objects(self, model: KGEI[KGEConfig], sr_batch: LongTensor2D):
        """Test score_objects method."""
        model.eval()
        scores = model.score_objects(sr_batch)
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (sr_batch.shape[0], model.config.num_entities)
        assert scores.dtype == torch.float32

    def test_score_subjects(self, model: KGEI[KGEConfig], ro_batch: LongTensor2D):
        """Test score_subjects method."""
        model.eval()
        scores = model.score_subjects(ro_batch)
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (ro_batch.shape[0], model.config.num_entities)
        assert scores.dtype == torch.float32

    def test_score_relations(self, model: KGEI[KGEConfig], so_batch: LongTensor2D):
        """Test score_relations method."""
        model.eval()
        scores = model.score_relations(so_batch)
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (so_batch.shape[0], model.config.num_relations)
        assert scores.dtype == torch.float32

    # ========== Cache Embeddings Tests ==========

    def test_set_cache_embeddings(self, model: KGEI[KGEConfig]):
        """Test setting cache embeddings flag."""
        model.set_cache_embeddings(True)
        assert model._cache_embeddings is True
        model.set_cache_embeddings(False)
        assert model._cache_embeddings is False

    # ========== Save/Load Tests ==========

    def test_save_model(self, model: KGEI[KGEConfig], tmp_path: Path):
        """Test saving model to disk."""
        save_dir = tmp_path / "model_save"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(save_dir)

        # Check that files were created
        assert save_dir.exists()
        assert (save_dir / "model.pt").exists()
        assert (save_dir / "config.json").exists()

    def test_load_config(self, model: KGEI[KGEConfig], tmp_path: Path):
        """Test loading config from saved model."""
        save_dir = tmp_path / "model_save"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(save_dir)

        loaded_config = model.__class__.load_config(save_dir)
        assert loaded_config.num_entities == model.config.num_entities
        assert loaded_config.num_relations == model.config.num_relations
        assert loaded_config.embedding_dim == model.config.embedding_dim

    def test_load_model(self, model: KGEI[KGEConfig], tmp_path: Path):
        """Test loading a saved model."""
        save_dir = tmp_path / "model_save"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(save_dir)

        loaded_model = model.__class__.load(save_dir, device="cpu")
        assert isinstance(loaded_model, model.__class__)
        assert loaded_model.config.num_entities == model.config.num_entities
        assert loaded_model.config.num_relations == model.config.num_relations
        assert loaded_model.config.embedding_dim == model.config.embedding_dim

    def test_save_load_roundtrip(
        self, model: KGEI[KGEConfig], sro_batch: LongTensor2D, tmp_path: Path
    ):
        """Test that saved and loaded model produces similar outputs."""
        model.eval()
        original_output = model(sro_batch)

        save_dir = tmp_path / "model_save"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(save_dir)
        loaded_model = model.__class__.load(save_dir, device="cpu")
        loaded_model.eval()

        loaded_output = loaded_model(sro_batch)
        # Outputs should have same shape
        assert original_output.scores.shape == loaded_output.scores.shape

    # ========== Save Embeddings Tests ==========

    def test_save_embeddings(self, model: KGEI[KGEConfig], tmp_path: Path):
        """Test saving entity and relation embeddings."""
        save_dir = tmp_path / "embeddings"
        model.eval()
        model.save_embeddings(save_dir, device="cpu")

        # Check that files were created
        assert save_dir.exists()
        assert (save_dir / "entities.pt").exists()
        assert (save_dir / "relations.pt").exists()

        # Check that embeddings can be loaded
        entity_embeddings = torch.load(save_dir / "entities.pt")
        relation_embeddings = torch.load(save_dir / "relations.pt")

        assert entity_embeddings.shape == (model.config.num_entities, model.config.embedding_dim)
        assert relation_embeddings.shape == (
            model.config.num_relations,
            model.config.embedding_dim,
        )

    # ========== Scaler Tests ==========

    def test_has_scaler_initially_false(self, model: KGEI[KGEConfig]):
        """Test that model has no scaler initially."""
        assert not model.has_scaler()

    def test_normalize_scores_without_scaler_raises(
        self, model: KGEI[KGEConfig], sro_batch: LongTensor2D
    ):
        """Test that normalize_scores raises error when no scaler is set."""
        model.eval()
        scores = model.score(sro_batch)
        with pytest.raises(ValueError, match="No score scaler set"):
            model.normalize_scores(scores)

    def test_fit_score_scaler(self, model: KGEI[KGEConfig]):
        """Test fitting a score scaler from a dataset."""
        # Create a simple dataset
        num_samples = 32
        sro_data = torch.randint(
            0,
            min(model.config.num_entities, model.config.num_relations),
            (num_samples, 3),
            dtype=torch.long,
        )
        dataset = TensorDataset(sro_data)

        model.eval()
        model.fit_score_scaler_from_dataset(dataset, json_path=None)

        assert model.has_scaler()

    def test_load_score_scaler(self, model: KGEI[KGEConfig], tmp_path: Path):
        """Test loading a score scaler from file."""
        # First fit and save a scaler
        num_samples = 32
        sro_data = torch.randint(
            0,
            min(model.config.num_entities, model.config.num_relations),
            (num_samples, 3),
            dtype=torch.long,
        )
        dataset = TensorDataset(sro_data)

        scaler_path = tmp_path / "scaler.json"
        model.eval()
        model.fit_score_scaler_from_dataset(dataset, json_path=str(scaler_path))

        # Create a new model and load the scaler
        new_model = model.__class__(model.config)
        success = new_model.load_score_scaler(str(scaler_path))

        assert success
        assert new_model.has_scaler()

    def test_normalize_scores_with_scaler(self, model: KGEI[KGEConfig], sro_batch: LongTensor2D):
        """Test normalizing scores with a fitted scaler."""
        # Fit a scaler
        num_samples = 32
        sro_data = torch.randint(
            0,
            min(model.config.num_entities, model.config.num_relations),
            (num_samples, 3),
            dtype=torch.long,
        )
        dataset = TensorDataset(sro_data)

        model.eval()
        model.fit_score_scaler_from_dataset(dataset, json_path=None)

        # Test normalization
        scores = model.score(sro_batch)
        normalized = model.normalize_scores(scores)

        assert normalized.shape == scores.shape
        assert normalized.dtype == torch.float32
        # Normalized scores should typically be in [0, 1] for sigmoid scaler
        assert torch.all(normalized >= 0) and torch.all(normalized <= 1)

    # ========== Parameters Tests ==========

    def test_parameters_iterable(self, model: KGEI[KGEConfig]):
        """Test that parameters() returns an iterable."""
        params = model.parameters()
        assert hasattr(params, "__iter__")
        # Should be able to convert to list (may be empty for some models)
        list(params)  # Should not raise

    # ========== Edge Cases ==========

    def test_empty_batch(self, model: KGEI[KGEConfig]):
        """Test handling of empty batch."""
        empty_batch = torch.empty((0, 3), dtype=torch.long)
        model.eval()
        output = model(empty_batch)
        assert output is not None
        assert output.scores.shape[0] == 0

    def test_single_triple(self, model: KGEI[KGEConfig]):
        """Test handling of single triple."""
        single_triple = torch.randint(
            0,
            min(model.config.num_entities, model.config.num_relations),
            (1, 3),
            dtype=torch.long,
        )
        model.eval()
        output = model(single_triple)
        assert output.scores.shape[0] == 1

    def test_large_batch(self, model: KGEI[KGEConfig]):
        """Test handling of large batch."""
        large_batch = torch.randint(
            0,
            min(model.config.num_entities, model.config.num_relations),
            (1024, 3),
            dtype=torch.long,
        )
        model.eval()
        output = model(large_batch)
        assert output.scores.shape[0] == 1024
