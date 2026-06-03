from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from contextualization.core.entities.config.publication_vector_database import (
    MilvusPublicationVectorDatabaseConfig,
)
from contextualization.core.entities.vector_database import (
    Metadata,
    VectorType,
    VectorWithMetadata,
)
from contextualization.impl.publication_vector_database.milvus import (
    MilvusPublicationVectorDatabase,
)

if TYPE_CHECKING:
    from contextualization.core.entities.types import Vector


@pytest.mark.milvus
class TestMilvusPublicationVectorDatabase:
    def test_init(self, milvus_connection_info):
        config = MilvusPublicationVectorDatabaseConfig(dimension=16, **milvus_connection_info)
        db = MilvusPublicationVectorDatabase(config)

        client = db._client
        indexes = client.list_indexes(collection_name=milvus_connection_info["collection_name"])
        assert "vector" in indexes
        assert "publication_id" in indexes
        assert "text_sparse" in indexes

    @pytest.mark.parametrize("dimension", [16])
    @pytest.mark.parametrize("db_vector_type", list(VectorType))
    @pytest.mark.parametrize("input_vector_type", ["list", "np", "pt"])
    def test(self, milvus_connection_info, dimension, db_vector_type, input_vector_type):
        config = MilvusPublicationVectorDatabaseConfig(
            dimension=dimension, vector_type=db_vector_type, **milvus_connection_info
        )
        vector_db = MilvusPublicationVectorDatabase(config)

        input_vectors: list[Vector]

        input_vectors = [
            [0.0] * dimension,
            [1.0] * dimension,
            [2.5] * dimension,
            [10.0] * dimension,
        ]
        metadata_list = [
            Metadata(publication_id="pub_1", chunk_index=1, num_chunks=2, text="pub 1 chunk 1"),
            Metadata(publication_id="pub_2", chunk_index=1, num_chunks=3, text="pub 2 chunk 1"),
            Metadata(publication_id="pub_2", chunk_index=2, num_chunks=3, text="pub 2 chunk 2"),
            Metadata(publication_id="pub_2", chunk_index=3, num_chunks=3, text="pub 2 chunk 3"),
        ]

        input_vectors_np = [np.asarray(v, dtype=np.float32) for v in input_vectors]

        if input_vector_type == "np":
            input_vectors = [np.asarray(v, dtype=np.float32) for v in input_vectors]
        elif input_vector_type == "pt":
            input_vectors = [torch.tensor(v, dtype=torch.float32) for v in input_vectors]

        input_data = [
            VectorWithMetadata(vector=v, metadata=metadata)
            for v, metadata in zip(input_vectors, metadata_list, strict=True)
        ]

        vector_db.insert(data=input_data)
        vector_db._client.flush(collection_name=milvus_connection_info["collection_name"])

        assert vector_db.exists(publication_id="pub_1")
        assert vector_db.exists(publication_id="pub_2")
        assert vector_db.exists(publication_id="pub_1", chunk_index=1)
        assert not vector_db.exists(publication_id="pub_1", chunk_index=2)
        assert vector_db.exists(publication_id="pub_2", chunk_index=1)
        assert vector_db.exists(publication_id="pub_2", chunk_index=2)
        assert vector_db.exists(publication_id="pub_2", chunk_index=3)

        retrieved_data_1 = vector_db.get_by_publication_ids(["pub_1"])
        retrieved_data_all = vector_db.get_by_publication_ids(["pub_1", "pub_2"], include_text=True)

        assert len(retrieved_data_1) == 1
        assert len(retrieved_data_all) == 4

        assert np.all(np.less(np.abs(retrieved_data_1[0].vector - input_vectors_np[0]), 1e-3))
        assert retrieved_data_1[0].metadata.text is None

        for vector_with_metadata in retrieved_data_all:
            assert isinstance(vector_with_metadata.vector, np.ndarray)
            assert vector_with_metadata.metadata.publication_id
            assert vector_with_metadata.metadata.num_chunks
            assert vector_with_metadata.metadata.chunk_index
            assert vector_with_metadata.metadata.text

        metadata_list = []
        for metadata in vector_db.metadata_iterator():
            metadata_list.append(metadata)
            assert isinstance(metadata, Metadata)
        assert len(metadata_list) == 4

        statistics = vector_db.retrieve_statistics()
        assert "pub_1" not in statistics.finished
        assert "pub_1" in statistics.partial
        assert "pub_2" in statistics.finished
        assert "pub_2" not in statistics.partial
