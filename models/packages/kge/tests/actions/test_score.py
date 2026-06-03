import random
from pathlib import Path

import pandas as pd
import pytest
import torch
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from kge.common.actions.kge_inference_action import KGEInferenceActions
from kge.common.actions.kge_loader_action import KGEExperimentData, KGELoader
from kge.common.entities import (
    KGEScoreIndexRequest,
    KGEScoreRequest,
    KGEScoreResponse,
)
from kge.data_processor.base import KGDataProcessor
from kge.models.base import KGEI

load_dotenv()


class TestSettings(BaseSettings):
    """Test configuration settings loaded from .env.test file."""

    model_config = SettingsConfigDict(
        env_file=".env.test",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    experiment_folder: str
    model_ckpt_path: str = "seed_0/model_checkpoint/last.ckpt"
    model_ckpt_is_lightning: bool = True
    score_scaler_json_path: str | None = None
    device: str = "cuda"

    @classmethod
    def create(cls) -> "TestSettings":
        """Factory method to create settings instance."""
        return cls()  # type: ignore


def _check_experiment_folder() -> bool:
    """Check if experiment folder exists and is accessible."""
    try:
        settings = TestSettings()  # type: ignore[call-arg]
        exp_path = Path(settings.experiment_folder)
        return exp_path.exists() and exp_path.is_dir()
    except Exception:
        return False


if not _check_experiment_folder():
    pytest.skip(
        "Experiment folder not found or test settings unavailable. Skipping integration tests.",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def config() -> TestSettings:
    """Fixture to provide test configuration."""
    return TestSettings.create()


@pytest.fixture(scope="module")
def kge_bundle(config: TestSettings) -> KGEExperimentData:
    return KGELoader.load_experiment(
        experiment_folder=config.experiment_folder,
        model_ckpt_path=config.model_ckpt_path,
        model_ckpt_is_lightning=config.model_ckpt_is_lightning,
        score_scaler_json_path=config.score_scaler_json_path,
        device=config.device,
    )


@pytest.fixture(scope="module")
def kge_model(kge_bundle: KGEExperimentData) -> KGEI:
    """Fixture to provide the KGE model from the bundle."""
    return kge_bundle.model


@pytest.fixture(scope="module")
def data_processor(kge_bundle: KGEExperimentData) -> KGDataProcessor:
    """Fixture to provide the data processor from the bundle."""
    return kge_bundle.data_processor


@pytest.fixture(scope="module")
def device(config: TestSettings) -> str:
    """Fixture to provide the device configuration."""
    return config.device


@pytest.fixture
def sample_entities(data_processor: KGDataProcessor) -> list[str]:
    """Provide sample entity IDs from the real dataset."""
    kg = data_processor.kg
    all_entities = kg.entity_mapping.get_ids()
    return random.sample(all_entities, min(5, len(all_entities)))


@pytest.fixture
def sample_relations(data_processor: KGDataProcessor) -> list[str]:
    """Provide sample relation IDs from the real dataset."""
    kg = data_processor.kg
    all_relations = kg.relation_mapping.get_ids()
    return random.sample(all_relations, min(3, len(all_relations)))


class TestScoreIndexIntegration:
    """Integration tests for score method."""

    def test_index_score_valid_fact(
        self, kge_model: KGEI, data_processor: KGDataProcessor, device: str
    ):
        """Test scoring with valid triple indices."""
        # Get some valid indices from the data processor
        kg = data_processor.kg
        entity_ids = kg.entity_mapping.get_ids()[:3]
        relation_ids = kg.relation_mapping.get_ids()[:2]

        # Convert to indices
        facts_index_list = [
            [
                kg.entity_mapping.id_to_index[entity_ids[0]],
                kg.relation_mapping.id_to_index[relation_ids[0]],
                kg.entity_mapping.id_to_index[entity_ids[1]],
            ],
            [
                kg.entity_mapping.id_to_index[entity_ids[1]],
                kg.relation_mapping.id_to_index[relation_ids[1]],
                kg.entity_mapping.id_to_index[entity_ids[2]],
            ],
        ]

        request = KGEScoreIndexRequest(
            facts_index_list=facts_index_list,
            normalize=False,
        )

        response = KGEInferenceActions.score_from_index(
            request=request,
            kge=kge_model,
            device=device,
        )

        assert isinstance(response, KGEScoreResponse)
        assert len(response.scores_list) == 2
        assert all(isinstance(s, float) for s in response.scores_list)

    def test_index_score_with_normalization(
        self, kge_model: KGEI, data_processor: KGDataProcessor, device: str
    ):
        """Test scoring with normalization enabled."""
        kg = data_processor.kg
        entity_ids = kg.entity_mapping.get_ids()[:3]
        relation_ids = kg.relation_mapping.get_ids()[:1]

        triple_indices = [
            [
                kg.entity_mapping.id_to_index[entity_ids[0]],
                kg.relation_mapping.id_to_index[relation_ids[0]],
                kg.entity_mapping.id_to_index[entity_ids[1]],
            ]
        ]

        request = KGEScoreIndexRequest(
            facts_index_list=triple_indices,
            normalize=True,
        )

        response = KGEInferenceActions.score_from_index(
            request=request,
            kge=kge_model,
            device=device,
        )

        assert len(response.scores_list) == 1
        if kge_model.has_scaler():
            # Normalized scores should typically be in a different range
            assert response.normalized_scores_list is not None
            assert isinstance(response.normalized_scores_list[0], float)

    def test_index_score_multiple_triples(
        self, kge_model: KGEI, data_processor: KGDataProcessor, device: str
    ):
        """Test scoring with multiple triples."""
        kg = data_processor.kg
        entity_ids = kg.entity_mapping.get_ids()[:5]
        relation_ids = kg.relation_mapping.get_ids()[:2]

        triple_indices = [
            [
                kg.entity_mapping.id_to_index[entity_ids[i]],
                kg.relation_mapping.id_to_index[relation_ids[i % len(relation_ids)]],
                kg.entity_mapping.id_to_index[entity_ids[(i + 1) % len(entity_ids)]],
            ]
            for i in range(5)
        ]

        request = KGEScoreIndexRequest(
            facts_index_list=triple_indices,
            normalize=False,
        )

        response = KGEInferenceActions.score_from_index(
            request=request,
            kge=kge_model,
            device=device,
        )

        assert len(response.scores_list) == 5
        assert all(isinstance(s, float) for s in response.scores_list)


class TestScoreIntegration:
    """Integration tests for score method."""

    def test_score_valid_fact(self, kge_model: KGEI, data_processor: KGDataProcessor, device: str):
        """Test scoring with valid triple indices."""
        # Get some valid indices from the data processor
        kg = data_processor.kg
        entity_ids = kg.entity_mapping.get_ids()[:3]
        relation_ids = kg.relation_mapping.get_ids()[:2]

        # Convert to indices
        facts_list = [
            (entity_ids[0], relation_ids[0], entity_ids[1]),
            (
                entity_ids[1],
                relation_ids[1],
                entity_ids[2],
            ),
        ]

        request = KGEScoreRequest(
            facts_list=facts_list,
            normalize=False,
        )

        response = KGEInferenceActions.score(
            request=request,
            kge=kge_model,
            data_processing=data_processor,
            device=device,
        )

        assert isinstance(response, KGEScoreResponse)
        assert len(response.scores_list) == 2
        assert all(isinstance(s, float) for s in response.scores_list)

    def test_score_with_normalization(
        self, kge_model: KGEI, data_processor: KGDataProcessor, device: str
    ):
        """Test scoring with normalization enabled."""
        kg = data_processor.kg
        entity_ids = kg.entity_mapping.get_ids()[:3]
        relation_ids = kg.relation_mapping.get_ids()[:1]

        facts_list = [
            (entity_ids[0], relation_ids[0], entity_ids[1]),
        ]

        request = KGEScoreRequest(
            facts_list=facts_list,
            normalize=True,
        )

        response = KGEInferenceActions.score(
            request=request,
            kge=kge_model,
            data_processing=data_processor,
            device=device,
        )

        assert len(response.scores_list) == 1
        if kge_model.has_scaler():
            # Normalized scores should typically be in a different range
            assert response.normalized_scores_list is not None
            assert isinstance(response.normalized_scores_list[0], float)

    def test_score_multiple_triples(
        self, kge_model: KGEI, data_processor: KGDataProcessor, device: str
    ):
        """Test scoring with multiple triples."""
        kg = data_processor.kg
        entity_ids = kg.entity_mapping.get_ids()[:5]
        relation_ids = kg.relation_mapping.get_ids()[:2]

        facts = [
            (
                entity_ids[i],
                relation_ids[i % len(relation_ids)],
                entity_ids[(i + 1) % len(entity_ids)],
            )
            for i in range(5)
        ]

        request = KGEScoreRequest(
            facts_list=facts,
            normalize=False,
        )

        response = KGEInferenceActions.score(
            request=request,
            kge=kge_model,
            data_processing=data_processor,
            device=device,
        )

        assert len(response.scores_list) == 5
        assert all(isinstance(s, float) for s in response.scores_list)


class TestScoreFromEntityListIntegration:
    """Integration tests for score_from_entity_list method."""

    def test_score_from_entity_list_basic(
        self,
        kge_model: KGEI,
        data_processor: KGDataProcessor,
        sample_entities,
        device: str,
    ):
        """Test basic functionality with small entity list."""
        entity_list = sample_entities[:3]

        result = KGEInferenceActions.score_from_entity_list(
            entity_id_list=entity_list,
            kge=kge_model,
            data_processing=data_processor,
            top_k=10,
            device=device,
            batch_size=1024,
        )

        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) <= 10  # Should respect top_k

        # Verify columns
        expected_columns = {"subject", "relation", "object", "score"}
        assert set(result.columns) == expected_columns

        # Verify no self-loops
        for _, row in result.iterrows():
            assert row["subject"] != row["object"]

        # Verify scores are sorted descending
        scores = result["score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_score_from_entity_list_top_k(
        self,
        kge_model: KGEI,
        data_processor: KGDataProcessor,
        sample_entities,
        device: str,
    ):
        """Test that top_k parameter is respected."""
        entity_list = sample_entities[:4]

        result = KGEInferenceActions.score_from_entity_list(
            entity_id_list=entity_list,
            kge=kge_model,
            data_processing=data_processor,
            top_k=5,
            device=device,
        )

        assert len(result) <= 5

    def test_score_from_entity_list_small_batch(
        self,
        kge_model: KGEI,
        data_processor: KGDataProcessor,
        sample_entities,
        device: str,
    ):
        """Test with small batch size to ensure batching works correctly."""
        entity_list = sample_entities[:3]

        result = KGEInferenceActions.score_from_entity_list(
            entity_id_list=entity_list,
            kge=kge_model,
            data_processing=data_processor,
            top_k=10,
            device=device,
            batch_size=2,  # Small batch to test batching logic
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) <= 10

    def test_score_from_entity_list_scores_validity(
        self,
        kge_model: KGEI,
        data_processor: KGDataProcessor,
        sample_entities,
        device: str,
    ):
        """Test that all scores are valid numbers."""
        entity_list = sample_entities[:3]

        result = KGEInferenceActions.score_from_entity_list(
            entity_id_list=entity_list,
            kge=kge_model,
            data_processing=data_processor,
            top_k=5,
            device=device,
        )

        # All scores should be valid floats
        assert result["score"].dtype in [float, "float32", "float64"]
        assert not result["score"].isna().any()

    def test_score_from_entity_list_entities_in_result(
        self,
        kge_model: KGEI,
        data_processor: KGDataProcessor,
        sample_entities,
        device: str,
    ):
        """Test that result only contains entities from the input list."""
        entity_list = sample_entities[:3]
        entity_set = set(entity_list)

        result = KGEInferenceActions.score_from_entity_list(
            entity_id_list=entity_list,
            kge=kge_model,
            data_processing=data_processor,
            top_k=10,
            device=device,
        )

        # All subjects and objects should be from input list
        for _, row in result.iterrows():
            assert row["subject"] in entity_set
            assert row["object"] in entity_set

    def test_score_from_entity_list_relations_validity(
        self,
        kge_model: KGEI,
        data_processor: KGDataProcessor,
        sample_entities,
        device: str,
    ):
        """Test that relations in result are valid."""
        entity_list = sample_entities[:3]
        valid_relations = set(data_processor.relation_list())

        result = KGEInferenceActions.score_from_entity_list(
            entity_id_list=entity_list,
            kge=kge_model,
            data_processing=data_processor,
            top_k=10,
            device=device,
        )

        # All relations should be valid
        for _, row in result.iterrows():
            assert row["relation"] in valid_relations


class TestScoreFromTensorIntegration:
    """Integration tests for score_from_tensor method."""

    def test_score_from_tensor_valid_fact(
        self, kge_model: KGEI, data_processor: KGDataProcessor, device: str
    ):
        """Test scoring with valid tensor indices."""
        kg = data_processor.kg
        entity_ids = kg.entity_mapping.get_ids()[:3]
        relation_ids = kg.relation_mapping.get_ids()[:2]

        sro_batch = torch.tensor(
            [
                [
                    kg.entity_mapping.id_to_index[entity_ids[0]],
                    kg.relation_mapping.id_to_index[relation_ids[0]],
                    kg.entity_mapping.id_to_index[entity_ids[1]],
                ],
                [
                    kg.entity_mapping.id_to_index[entity_ids[1]],
                    kg.relation_mapping.id_to_index[relation_ids[1]],
                    kg.entity_mapping.id_to_index[entity_ids[2]],
                ],
            ],
            dtype=torch.long,
        )

        response = KGEInferenceActions.score_from_tensor(
            sro_batch=sro_batch,
            kge=kge_model,
            normalize=False,
            device=device,
        )

        assert isinstance(response, KGEScoreResponse)
        assert len(response.scores_list) == 2
        assert all(isinstance(s, float) for s in response.scores_list)
        assert response.normalized_scores_list is None

    def test_score_from_tensor_with_normalization(
        self, kge_model: KGEI, data_processor: KGDataProcessor, device: str
    ):
        """Test scoring with normalization enabled."""
        kg = data_processor.kg
        entity_ids = kg.entity_mapping.get_ids()[:3]
        relation_ids = kg.relation_mapping.get_ids()[:1]

        sro_batch = torch.tensor(
            [
                [
                    kg.entity_mapping.id_to_index[entity_ids[0]],
                    kg.relation_mapping.id_to_index[relation_ids[0]],
                    kg.entity_mapping.id_to_index[entity_ids[1]],
                ]
            ],
            dtype=torch.long,
        )

        response = KGEInferenceActions.score_from_tensor(
            sro_batch=sro_batch,
            kge=kge_model,
            normalize=True,
            device=device,
        )

        assert len(response.scores_list) == 1
        assert all(isinstance(s, float) for s in response.scores_list)

        if kge_model.has_scaler():
            assert response.normalized_scores_list is not None
            assert len(response.normalized_scores_list) == 1
            assert isinstance(response.normalized_scores_list[0], float)

    def test_score_from_tensor_multiple_triples(
        self, kge_model: KGEI, data_processor: KGDataProcessor, device: str
    ):
        """Test scoring with multiple triples."""
        kg = data_processor.kg
        entity_ids = kg.entity_mapping.get_ids()[:5]
        relation_ids = kg.relation_mapping.get_ids()[:2]

        sro_batch = torch.tensor(
            [
                [
                    kg.entity_mapping.id_to_index[entity_ids[i]],
                    kg.relation_mapping.id_to_index[relation_ids[i % len(relation_ids)]],
                    kg.entity_mapping.id_to_index[entity_ids[(i + 1) % len(entity_ids)]],
                ]
                for i in range(5)
            ],
            dtype=torch.long,
        )

        response = KGEInferenceActions.score_from_tensor(
            sro_batch=sro_batch,
            kge=kge_model,
            normalize=False,
            device=device,
        )

        assert len(response.scores_list) == 5
        assert all(isinstance(s, float) for s in response.scores_list)

    def test_score_from_tensor_with_invalid_indices(
        self, kge_model: KGEI, data_processor: KGDataProcessor, device: str
    ):
        """Test scoring with invalid indices (marked with -1)."""
        kg = data_processor.kg
        entity_ids = kg.entity_mapping.get_ids()[:3]
        relation_ids = kg.relation_mapping.get_ids()[:1]

        sro_batch = torch.tensor(
            [
                [
                    kg.entity_mapping.id_to_index[entity_ids[0]],
                    kg.relation_mapping.id_to_index[relation_ids[0]],
                    kg.entity_mapping.id_to_index[entity_ids[1]],
                ],
                [-1, -1, -1],  # Invalid triple
                [
                    kg.entity_mapping.id_to_index[entity_ids[1]],
                    kg.relation_mapping.id_to_index[relation_ids[0]],
                    kg.entity_mapping.id_to_index[entity_ids[2]],
                ],
            ],
            dtype=torch.long,
        )

        response = KGEInferenceActions.score_from_tensor(
            sro_batch=sro_batch,
            kge=kge_model,
            normalize=False,
            device=device,
        )

        assert isinstance(response, KGEScoreResponse)
        assert len(response.scores_list) == 3
        assert all(isinstance(s, float) for s in response.scores_list)

        assert response.scores_list[0] != response.scores_list[1]
        assert response.scores_list[2] != response.scores_list[1]

    def test_score_from_tensor_partially_invalid_indices(
        self, kge_model: KGEI, data_processor: KGDataProcessor, device: str
    ):
        """Test scoring with partially invalid indices (some elements are -1)."""
        kg = data_processor.kg
        entity_ids = kg.entity_mapping.get_ids()[:2]
        relation_ids = kg.relation_mapping.get_ids()[:1]

        sro_batch = torch.tensor(
            [
                [
                    kg.entity_mapping.id_to_index[entity_ids[0]],
                    kg.relation_mapping.id_to_index[relation_ids[0]],
                    kg.entity_mapping.id_to_index[entity_ids[1]],
                ],
                [
                    kg.entity_mapping.id_to_index[entity_ids[0]],
                    -1,  # Invalid relation
                    kg.entity_mapping.id_to_index[entity_ids[1]],
                ],
            ],
            dtype=torch.long,
        )

        response = KGEInferenceActions.score_from_tensor(
            sro_batch=sro_batch,
            kge=kge_model,
            normalize=False,
            device=device,
        )

        assert len(response.scores_list) == 2
        assert all(isinstance(s, float) for s in response.scores_list)
        assert response.scores_list[0] != response.scores_list[1]

    def test_score_from_tensor_single_triple(
        self, kge_model: KGEI, data_processor: KGDataProcessor, device: str
    ):
        """Test scoring with a single triple."""
        kg = data_processor.kg
        entity_ids = kg.entity_mapping.get_ids()[:2]
        relation_ids = kg.relation_mapping.get_ids()[:1]

        sro_batch = torch.tensor(
            [
                [
                    kg.entity_mapping.id_to_index[entity_ids[0]],
                    kg.relation_mapping.id_to_index[relation_ids[0]],
                    kg.entity_mapping.id_to_index[entity_ids[1]],
                ]
            ],
            dtype=torch.long,
        )

        response = KGEInferenceActions.score_from_tensor(
            sro_batch=sro_batch,
            kge=kge_model,
            normalize=False,
            device=device,
        )

        assert len(response.scores_list) == 1
        assert isinstance(response.scores_list[0], float)
