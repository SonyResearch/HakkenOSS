from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from transformers import AutoConfig, AutoModel, AutoTokenizer

from contextualization.core.entities.config.publication_encoder import (
    LLMPublicationEncoderConfig,
)
from contextualization.core.entities.config.publication_vector_database import (
    InMemoryPublicationVectorDatabaseConfig,
)
from contextualization.core.entities.publication import Publication
from contextualization.core.entities.vector_database import Metadata
from contextualization.impl.publication_encoder import LLMPublicationEncoder
from contextualization.impl.publication_vector_database.in_memory import (
    InMemoryPublicationVectorDatabase,
)
from contextualization.impl.reference_reader import ParquetReferenceReader

if TYPE_CHECKING:
    from contextualization.core.contracts.reference_reader import ReferenceReader


@pytest.fixture
def reference_reader() -> "ReferenceReader":
    publications = [
        Publication(
            publication_id=f"pub{i}",
            year=2000 + i,
            title=f"pub title {i}",
            abstract=f"pub abstract {i}",
        )
        for i in range(10)
    ]
    publications.append(
        Publication(
            publication_id="pub_long",
            year=2020,
            title="long publication_encoder" * 100,
            abstract="long publication_encoder abstract" * 100,
        )
    )
    reference_reader = MagicMock(spec=ParquetReferenceReader)
    reference_reader.iter_publications = MagicMock(
        side_effect=lambda num_skips=0: iter(publications[num_skips:])
    )

    return reference_reader


@pytest.fixture
def publication_encoder(request, tmp_path) -> LLMPublicationEncoder:
    model_type = request.param
    cache_dir = tmp_path / "cache"

    if model_type == "qwen3":
        hf_config = AutoConfig.from_pretrained("Qwen/Qwen3-Embedding-0.6B", cache_dir=cache_dir)
        hf_config.hidden_size = 24
        hf_config.intermediate_size = 32
        hf_config.num_hidden_layers = 2
        hf_config.layer_types = hf_config.layer_types[: hf_config.num_hidden_layers]

        model = AutoModel.from_config(hf_config)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", cache_dir=cache_dir)

        model.save_pretrained(tmp_path)
        tokenizer.save_pretrained(tmp_path)
    else:
        raise ValueError(f"unknown model type {model_type}")

    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)

    config = LLMPublicationEncoderConfig(
        hf_model_name_or_path=str(tmp_path), max_length=64, overlap=16, hf_model_loading_kwargs={}
    )

    return LLMPublicationEncoder(config)


class TestLLMPublicationEncoder:
    @pytest.mark.parametrize("publication_encoder", ["qwen3"], indirect=True)
    @pytest.mark.parametrize("batch_size", [2])
    def test_encode_and_store_to_db(
        self,
        publication_encoder,
        reference_reader,
        batch_size,
    ):
        vector_db_config = InMemoryPublicationVectorDatabaseConfig()
        vector_db = InMemoryPublicationVectorDatabase(vector_db_config)

        publication_encoder.config.batch_size = batch_size
        publication_encoder.encode_and_store_to_db(
            reference_reader=reference_reader,
            publication_vector_database=vector_db,
            skip_existing=True,
        )
        assert vector_db.exists("pub_long", 1)

        for _ in publication_encoder.batch_iterator(
            reference_reader=reference_reader,
            batch_size=batch_size,
            publication_vector_database=vector_db,
            skip_existing=True,
        ):
            # Ensure no batch is left
            raise AssertionError()

    @pytest.mark.parametrize("publication_encoder", ["qwen3"], indirect=True)
    @pytest.mark.parametrize("max_length", [None, 128])
    @pytest.mark.parametrize("batch_size", [2, 6])
    def test_batch_iterator(self, publication_encoder, reference_reader, max_length, batch_size):
        it = publication_encoder.batch_iterator(
            reference_reader=reference_reader, batch_size=batch_size, skip_existing=False
        )

        batches = []
        for inputs_batch, metadata_batch in it:
            assert "input_ids" in inputs_batch
            assert "attention_mask" in inputs_batch
            assert inputs_batch["input_ids"].shape[0] == len(metadata_batch)
            assert isinstance(metadata_batch[0], Metadata)
            batches.append(inputs_batch)

        for i, inputs_batch in enumerate(batches):
            assert inputs_batch["input_ids"].shape == inputs_batch["attention_mask"].shape
            decoded_inputs = publication_encoder.tokenizer.batch_decode(
                inputs_batch["input_ids"], skip_special_tokens=True
            )
            for decoded_input in decoded_inputs:
                assert publication_encoder.config.title_prefix in decoded_input
                assert publication_encoder.config.abstract_prefix in decoded_input
            if max_length is not None:
                assert inputs_batch["input_ids"].shape[1] <= max_length
            if i < len(batches) - 1:
                assert inputs_batch["input_ids"].shape[0] == batch_size
            else:
                assert inputs_batch["input_ids"].shape[0] <= batch_size

    @pytest.mark.parametrize("publication_encoder", ["qwen3"], indirect=True)
    @pytest.mark.parametrize("batch_size", [8])
    def test_batch_iterator_with_skip(
        self,
        publication_encoder,
        reference_reader,
        batch_size,
    ):
        vector_db_config = InMemoryPublicationVectorDatabaseConfig()
        vector_db = InMemoryPublicationVectorDatabase(vector_db_config)

        publication_encoder.config.batch_size = batch_size
        publication_encoder.encode_and_store_to_db(
            reference_reader=reference_reader,
            publication_vector_database=vector_db,
            skip_existing=True,
        )

        it = publication_encoder.batch_iterator(
            reference_reader=reference_reader,
            batch_size=batch_size,
            publication_vector_database=vector_db,
            skip_existing=True,
        )

        for _ in it:
            raise AssertionError()

    @pytest.mark.parametrize("publication_encoder", ["qwen3"], indirect=True)
    @pytest.mark.parametrize("batch_size", [8])
    def test_encode(self, publication_encoder, reference_reader, batch_size):
        it = publication_encoder.batch_iterator(
            reference_reader=reference_reader, batch_size=batch_size, skip_existing=False
        )

        batch, _ = next(it)
        batch_embeddings = publication_encoder._encode(**batch)
        assert batch_embeddings.ndim == 2
        assert batch_embeddings.shape[0] == batch["input_ids"].shape[0]
        assert batch_embeddings.shape[1] == publication_encoder.model.config.hidden_size
