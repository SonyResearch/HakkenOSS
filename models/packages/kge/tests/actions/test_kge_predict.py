import random
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from kge.common.actions.kge_inference_action import KGEInferenceActions
from kge.common.actions.kge_loader_action import KGEExperimentData, KGELoader
from kge.common.entities import (
    KGEPredictRequest,
    KGEPredictResponse,
    KGEScoreIndexRequest,
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


class TestEntityPairToTriplesIntegration:
    """Integration tests for entity_pair_to_triples with real data."""

    def test_entity_pair_to_triples_real_data(self, data_processor: KGDataProcessor):
        """Test triple generation with real entity and relation IDs."""
        kg = data_processor.kg

        all_entities = kg.entity_mapping.get_ids()
        all_relations = kg.relation_mapping.get_ids()

        subject_ids = random.sample(all_entities, min(2, len(all_entities)))
        relation_ids = random.sample(all_relations, min(2, len(all_relations)))
        available_objects = [e for e in all_entities if e not in subject_ids]
        if len(available_objects) >= 2:
            object_ids = random.sample(available_objects, 2)
        else:
            object_ids = random.sample(all_entities, min(2, len(all_entities)))

        result = KGEInferenceActions.entity_pair_to_triples(
            subject_ids=subject_ids,
            relation_ids=relation_ids,
            object_ids=object_ids,
        )

        assert all(isinstance(t, tuple) and len(t) == 3 for t in result)

        # Verify all combinations are generated
        expected_combinations = {
            (subject_ids[0], relation_ids[0], object_ids[0]),
            (subject_ids[0], relation_ids[1], object_ids[0]),
            (subject_ids[1], relation_ids[0], object_ids[1]),
            (subject_ids[1], relation_ids[1], object_ids[1]),
        }
        assert set(result) == expected_combinations

    def test_entity_pair_to_triples_single_relation(self, sample_entities: list[str]):
        """Test triple generation with a single relation."""
        result = KGEInferenceActions.entity_pair_to_triples(
            subject_ids=[sample_entities[0]],
            relation_ids=["rel1"],
            object_ids=[sample_entities[1]],
        )

        assert len(result) == 1
        assert result[0] == (sample_entities[0], "rel1", sample_entities[1])


class TestPredictBatchIntegration:
    """Integration tests for predict_batch method."""

    def test_predict_batch_valid_triples(
        self,
        config: TestSettings,
        kge_model: KGEI,
        data_processor: KGDataProcessor,
        device: str,
    ):
        """Test predict_batch with valid triples from the knowledge graph."""
        kg = data_processor.kg
        all_entities = kg.entity_mapping.get_ids()
        all_relations = kg.relation_mapping.get_ids()

        # Create valid triples
        num_pairs = 3
        num_relations = 2
        triples_list = []
        for i in range(num_pairs):
            for j in range(num_relations):
                triples_list.append(
                    (
                        all_entities[i % len(all_entities)],
                        all_relations[j % len(all_relations)],
                        all_entities[(i + 1) % len(all_entities)],
                    )
                )

        result = KGEInferenceActions.predict_batch(
            triples_list=triples_list,
            num_pairs=num_pairs,
            num_relations=num_relations,
            kge=kge_model,
            data_processing=data_processor,
            device=device,
        )

        # Verify structure
        assert "relations_scores" in result
        assert "relations_probs" in result
        assert isinstance(result["relations_scores"], list)
        assert len(result["relations_scores"]) == num_pairs

        # Each pair should have scores for all relations
        for scores in result["relations_scores"]:
            assert len(scores) == num_relations
            assert all(isinstance(s, float) for s in scores)

        # If model has scaler, verify probs
        if config.score_scaler_json_path is not None:
            assert result["relations_probs"] is not None
            assert len(result["relations_probs"]) == num_pairs
            for probs in result["relations_probs"]:
                assert len(probs) == num_relations
                assert all(isinstance(p, float) for p in probs)

    def test_predict_batch_invalid_triples(
        self, kge_model: KGEI, data_processor: KGDataProcessor, device: str
    ):
        """Test predict_batch with invalid/missing entities."""
        triples_list = [
            ("invalid_entity_1", "invalid_relation", "invalid_entity_2"),
            ("invalid_entity_3", "invalid_relation", "invalid_entity_4"),
        ]

        result = KGEInferenceActions.predict_batch(
            triples_list=triples_list,
            num_pairs=2,
            num_relations=1,
            kge=kge_model,
            data_processing=data_processor,
            device=device,
        )

        # Should return default scores for invalid triples
        assert len(result["relations_scores"]) == 2
        for scores in result["relations_scores"]:
            assert len(scores) == 1
            # Default score should be -1000.0
            assert scores[0] == -1000.0

    def test_predict_batch_mixed_validity(
        self, kge_model: KGEI, data_processor: KGDataProcessor, device: str
    ):
        """Test predict_batch with mix of valid and invalid triples."""
        kg = data_processor.kg
        all_entities = kg.entity_mapping.get_ids()
        all_relations = kg.relation_mapping.get_ids()

        triples_list = [
            (all_entities[0], all_relations[0], all_entities[1]),  # Valid
            (all_entities[0], "invalid_relation", "invalid_entity2"),  # Invalid
        ]

        result = KGEInferenceActions.predict_batch(
            triples_list=triples_list,
            num_pairs=2,
            num_relations=1,
            kge=kge_model,
            data_processing=data_processor,
            device=device,
        )

        assert len(result["relations_scores"]) == 2
        # First should have a real score
        assert result["relations_scores"][0][0] != -1000.0
        # Second should have default score
        assert result["relations_scores"][1][0] == -1000.0


class TestPredictIntegration:
    """Integration tests for predict method."""

    def test_predict_single_batch(
        self,
        config: TestSettings,
        kge_model: KGEI,
        data_processor: KGDataProcessor,
        sample_entities,
        sample_relations,
        device: str,
    ):
        """Test predict with a small batch that fits in single processing."""
        request = KGEPredictRequest(
            subject_id_list=[sample_entities[0], sample_entities[1]],
            relation_id_list=sample_relations[:2],
            object_id_list=[sample_entities[2], sample_entities[3]],
            inference_config=None,
        )

        response = KGEInferenceActions.predict(
            request=request,
            kge=kge_model,
            data_processing=data_processor,
            device=device,
        )

        # Verify response structure
        assert isinstance(response, KGEPredictResponse)
        assert response.relations_ids == sample_relations[:2]

        assert response.relations_scores is not None
        assert len(response.relations_scores) == 2  # Two pairs
        assert all(len(scores) == 2 for scores in response.relations_scores)  # Two relations each
        if config.score_scaler_json_path is not None:
            assert response.relations_probs is not None
            assert len(response.relations_probs) == 2

    def test_predict_multiple_batches(
        self,
        kge_model: KGEI,
        data_processor: KGDataProcessor,
        sample_entities,
        sample_relations,
        device: str,
    ):
        """Test predict with multiple batches using small batch_size."""
        # Create request with 100 pairs
        num_pairs = 100
        request = KGEPredictRequest(
            subject_id_list=[sample_entities[i % len(sample_entities)] for i in range(num_pairs)],
            relation_id_list=sample_relations[:2],
            object_id_list=[
                sample_entities[(i + 1) % len(sample_entities)] for i in range(num_pairs)
            ],
            inference_config={"batch_size": 3},  # Force multiple batches
        )

        response = KGEInferenceActions.predict(
            request=request,
            kge=kge_model,
            data_processing=data_processor,
            device=device,
        )

        # Verify response
        assert response.relations_scores is not None
        assert len(response.relations_scores) == num_pairs
        assert all(len(scores) == 2 for scores in response.relations_scores)

    def test_predict_no_relations_raises_error(
        self,
        kge_model: KGEI,
        data_processor: KGDataProcessor,
        sample_entities,
        device: str,
    ):
        """Test that predict raises error when relation_id_list is None."""
        request = KGEPredictRequest(
            subject_id_list=[sample_entities[0]],
            relation_id_list=None,
            object_id_list=[sample_entities[1]],
        )

        with pytest.raises(NotImplementedError):
            KGEInferenceActions.predict(
                request=request,
                kge=kge_model,
                data_processing=data_processor,
                device=device,
            )

    def test_predict_custom_batch_size(
        self,
        kge_model: KGEI,
        data_processor: KGDataProcessor,
        sample_entities,
        sample_relations,
        device: str,
    ):
        """Test predict with custom batch size in inference_config."""
        request = KGEPredictRequest(
            subject_id_list=[sample_entities[0], sample_entities[1]],
            relation_id_list=sample_relations[:2],
            object_id_list=[sample_entities[2], sample_entities[3]],
            inference_config={"batch_size": 1},
        )

        response = KGEInferenceActions.predict(
            request=request,
            kge=kge_model,
            data_processing=data_processor,
            device=device,
        )
        assert response.relations_scores is not None
        assert len(response.relations_scores) == 2
        assert all(len(scores) == 2 for scores in response.relations_scores)


class TestEndToEndIntegration:
    """End-to-end integration tests combining multiple methods."""

    def test_predict_and_score_consistency(
        self,
        kge_model: KGEI,
        data_processor: KGDataProcessor,
        sample_entities,
        sample_relations,
        device: str,
    ):
        """Test that predict and score methods give consistent results."""
        kg = data_processor.kg

        # Use predict to get scores
        predict_request = KGEPredictRequest(
            subject_id_list=[sample_entities[0]],
            relation_id_list=[sample_relations[0]],
            object_id_list=[sample_entities[1]],
        )

        predict_response = KGEInferenceActions.predict(
            request=predict_request,
            kge=kge_model,
            data_processing=data_processor,
            device=device,
        )
        assert predict_response.relations_scores is not None

        predicted_score = predict_response.relations_scores[0][0]

        # Use score method with indices
        triple_indices = [
            [
                kg.entity_mapping.id_to_index[sample_entities[0]],
                kg.relation_mapping.id_to_index[sample_relations[0]],
                kg.entity_mapping.id_to_index[sample_entities[1]],
            ]
        ]

        score_request = KGEScoreIndexRequest(
            facts_index_list=triple_indices,
            normalize=False,
        )

        score_response = KGEInferenceActions.score_from_index(
            request=score_request,
            kge=kge_model,
            device=device,
        )

        direct_score = score_response.scores_list[0]

        # Scores should be very close (allowing for floating point differences)
        assert abs(predicted_score - direct_score) < 1e-5

    def test_full_workflow(
        self,
        kge_model: KGEI,
        data_processor: KGDataProcessor,
        sample_entities,
        sample_relations,
        device: str,
    ):
        """Test a complete workflow: generate triples -> predict -> verify."""
        # Generate triples
        triples = KGEInferenceActions.entity_pair_to_triples(
            subject_ids=sample_entities[:2],
            relation_ids=sample_relations[:2],
            object_ids=sample_entities[2:4],
        )

        assert len(triples) == 4

        # Predict scores
        predict_request = KGEPredictRequest(
            subject_id_list=sample_entities[:2],
            relation_id_list=sample_relations[:2],
            object_id_list=sample_entities[2:4],
        )

        response = KGEInferenceActions.predict(
            request=predict_request,
            kge=kge_model,
            data_processing=data_processor,
            device=device,
        )

        # Verify all scores are present
        assert response.relations_scores is not None
        assert len(response.relations_scores) == 2
        assert all(len(scores) == 2 for scores in response.relations_scores)
        assert all(isinstance(s, float) for scores in response.relations_scores for s in scores)
