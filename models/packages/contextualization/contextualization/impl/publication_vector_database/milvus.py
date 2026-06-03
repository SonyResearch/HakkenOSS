import logging
from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING, Final
from uuid import uuid4

import ml_dtypes
import numpy as np
import torch
import torch.distributed as dist
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    connections,
)
from pymilvus.milvus_client import IndexParams
from tqdm.auto import tqdm

from contextualization.core.contracts.publication_vector_database import (
    PublicationVectorDatabase,
)
from contextualization.core.entities.config.publication_vector_database import (
    MilvusPublicationVectorDatabaseConfig,
)
from contextualization.core.entities.vector_database import (
    Metadata,
    VectorDatabaseStatistics,
    VectorType,
    VectorWithMetadata,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from contextualization.core.entities.publication import PublicationId
    from contextualization.core.entities.types import NumpyVector, Vector

logger = logging.getLogger(__name__)


class MilvusPublicationVectorDatabase(
    PublicationVectorDatabase[MilvusPublicationVectorDatabaseConfig]
):
    ID_FIELD_NAME: Final[str] = "id"
    PUBLICATION_ID_FIELD_NAME: Final[str] = "publication_id"
    VECTOR_FIELD_NAME: Final[str] = "vector"
    TEXT_FIELD_NAME: Final[str] = "text"
    TEXT_SPARSE_FIELD_NAME: Final[str] = "text_sparse"
    CHUNK_INDEX_FIELD_NAME: Final[str] = "chunk_index"
    NUM_CHUNKS_FIELD_NAME: Final[str] = "num_chunks"

    PUBLICATION_ID_MAX_LENGTH: Final[int] = 32
    TEXT_MAX_LENGTH: Final[int] = 32768

    def __init__(self, config: MilvusPublicationVectorDatabaseConfig) -> None:
        super().__init__(config)

        self._client = MilvusClient(
            uri=self.config.uri,
            user=self.config.user,
            password=self.config.password,
            db_name=self.config.db_name,
            token=self.config.token,
            timeout=self.config.timeout,
        )
        self._create_collection()

        if dist.is_initialized():
            # Make sure that collection creation is finished
            dist.barrier()

    def _create_index_params(self) -> IndexParams:
        index_params = IndexParams()
        if not self.config.uri.endswith(".db") or self.config.vector_type == "float":
            # Milvus Lite doesn't support fp16 and bf16 indexing
            index_params.add_index(
                field_name=self.VECTOR_FIELD_NAME,
                index_type="AUTOINDEX",
                metric_type=self.config.metric_type.value,
            )
        index_params.add_index(
            field_name=self.TEXT_SPARSE_FIELD_NAME, index_type="AUTOINDEX", metric_type="BM25"
        )
        index_params.add_index(field_name=self.PUBLICATION_ID_FIELD_NAME, index_type="AUTOINDEX")

        return index_params

    def _create_schema(self) -> CollectionSchema:
        if self.config.vector_type == VectorType.FLOAT:
            milvus_vector_type = DataType.FLOAT_VECTOR
        elif self.config.vector_type == VectorType.FLOAT16:
            milvus_vector_type = DataType.FLOAT16_VECTOR
        elif self.config.vector_type == VectorType.BFLOAT16:
            milvus_vector_type = DataType.BFLOAT16_VECTOR
        else:
            raise ValueError(f"Unknown vector type: {self.config.vector_type}")

        # Let ID be auto increment, and then manage chunk id by publication_id & chunk_index fields
        # Additionally add text field for BM25 etc.
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
        )
        schema.add_field(
            field_name=self.VECTOR_FIELD_NAME,
            datatype=milvus_vector_type,
            dim=self.config.dimension,
        )
        schema.add_field(
            field_name=self.PUBLICATION_ID_FIELD_NAME,
            datatype=DataType.VARCHAR,
            max_length=self.PUBLICATION_ID_MAX_LENGTH,
        )
        schema.add_field(
            field_name=self.CHUNK_INDEX_FIELD_NAME, datatype=DataType.INT32, nullable=True
        )
        schema.add_field(
            field_name=self.NUM_CHUNKS_FIELD_NAME, datatype=DataType.INT32, nullable=True
        )
        schema.add_field(
            field_name=self.TEXT_FIELD_NAME,
            datatype=DataType.VARCHAR,
            max_length=self.TEXT_MAX_LENGTH,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name=self.TEXT_SPARSE_FIELD_NAME, datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["text_sparse"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        return schema

    def _create_collection(self) -> None:
        # In a distributed setup, create collection only from rank 0
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank != 0:
                return

        if self._client.has_collection(self.config.collection_name):
            return

        schema = self._create_schema()
        index_params = self._create_index_params()

        self._client.create_collection(
            collection_name=self.config.collection_name, schema=schema, index_params=index_params
        )

    def retrieve_statistics(self) -> VectorDatabaseStatistics:
        complete_publication_ids: list[PublicationId] = []
        chunk_existence: dict[PublicationId, set[int]] = defaultdict(lambda: set())

        num_rows = self._client.get_collection_stats(self.config.collection_name)["row_count"]

        for metadata in tqdm(self.metadata_iterator(include_text=False), total=num_rows):
            pub_id = metadata.publication_id
            num_chunks = metadata.num_chunks
            chunk_index = metadata.chunk_index
            chunk_existence[pub_id].add(chunk_index)
            if len(chunk_existence[pub_id]) == num_chunks:
                complete_publication_ids.append(pub_id)
                del chunk_existence[pub_id]

        return VectorDatabaseStatistics(
            finished=set(complete_publication_ids), partial=set(chunk_existence.keys())
        )

    def metadata_iterator(self, include_text: bool = False) -> "Iterator[Metadata]":
        connection_alias = uuid4().hex
        connections.connect(
            alias=connection_alias,
            uri=self.config.uri,
            user=self.config.user,
            password=self.config.password,
            db_name=self.config.db_name,
            token=self.config.token,
            timeout=self.config.timeout,
        )

        collection = Collection(self.config.collection_name, using=connection_alias)
        collection.flush()

        output_fields = [
            self.PUBLICATION_ID_FIELD_NAME,
            self.CHUNK_INDEX_FIELD_NAME,
            self.NUM_CHUNKS_FIELD_NAME,
        ]
        if include_text:
            output_fields.append(self.TEXT_FIELD_NAME)

        iterator = collection.query_iterator(output_fields=output_fields)
        while True:
            res = iterator.next()
            if not res:
                iterator.close()
                break

            metadata_list = [
                Metadata(
                    publication_id=row[self.PUBLICATION_ID_FIELD_NAME],
                    chunk_index=row[self.CHUNK_INDEX_FIELD_NAME],
                    num_chunks=row[self.NUM_CHUNKS_FIELD_NAME],
                    text=row.get(self.TEXT_FIELD_NAME, None),
                )
                for row in res
            ]
            yield from metadata_list

        connections.disconnect(connection_alias)

    def _existing_chunk_indices(self, publication_id: "PublicationId") -> set[int]:
        filter_condition = f"{self.PUBLICATION_ID_FIELD_NAME} == '{publication_id}'"

        rows = self._client.query(
            self.config.collection_name,
            filter=filter_condition,
            output_fields=[self.CHUNK_INDEX_FIELD_NAME],
        )
        chunk_indices = set()
        for row in rows:
            chunk_indices.add(row[self.CHUNK_INDEX_FIELD_NAME])

        return chunk_indices

    def exists(self, publication_id: "PublicationId", chunk_index: int | None = None) -> bool:
        chunk_indices = self._existing_chunk_indices(publication_id)

        if chunk_index is None:
            return bool(chunk_indices)
        return chunk_index in chunk_indices

    def get_by_publication_ids(
        self,
        publication_ids: "Sequence[PublicationId]",
        include_text: bool = False,
    ) -> "Sequence[VectorWithMetadata]":
        filter_condition = f"{self.PUBLICATION_ID_FIELD_NAME} in {list(publication_ids)!s}"

        output_fields = [
            self.PUBLICATION_ID_FIELD_NAME,
            self.CHUNK_INDEX_FIELD_NAME,
            self.NUM_CHUNKS_FIELD_NAME,
            self.VECTOR_FIELD_NAME,
        ]
        if include_text:
            output_fields.append(self.TEXT_FIELD_NAME)

        rows = self._client.query(
            self.config.collection_name, filter=filter_condition, output_fields=output_fields
        )

        results: list[VectorWithMetadata] = []
        for row in rows:
            metadata = Metadata(
                publication_id=row[self.PUBLICATION_ID_FIELD_NAME],
                chunk_index=row[self.CHUNK_INDEX_FIELD_NAME],
                num_chunks=row[self.NUM_CHUNKS_FIELD_NAME],
                text=row.get(self.TEXT_FIELD_NAME, None),
            )

            vector = row[self.VECTOR_FIELD_NAME]
            if isinstance(vector[0], bytes):
                vector = np.frombuffer(vector[0], dtype=self._vector_numpy_dtype)
            else:
                vector = np.array(vector, dtype=self._vector_numpy_dtype)

            vector_with_metadata = VectorWithMetadata(vector=vector, metadata=metadata)
            results.append(vector_with_metadata)

        return results

    def insert(self, data: "Sequence[VectorWithMetadata]") -> None:
        milvus_insert_data = []
        for vector_with_metadata in data:
            data_dict = {
                self.VECTOR_FIELD_NAME: self._convert_vector_to_array_for_insertion(
                    vector_with_metadata.vector
                ),
                self.PUBLICATION_ID_FIELD_NAME: vector_with_metadata.metadata.publication_id,
                self.CHUNK_INDEX_FIELD_NAME: vector_with_metadata.metadata.chunk_index,
                self.NUM_CHUNKS_FIELD_NAME: vector_with_metadata.metadata.num_chunks,
                self.TEXT_FIELD_NAME: vector_with_metadata.metadata.text,
            }
            milvus_insert_data.append(data_dict)

        self._client.insert(collection_name=self.config.collection_name, data=milvus_insert_data)

    @cached_property
    def _vector_torch_dtype(self) -> torch.dtype:
        dtype: torch.dtype
        if self.config.vector_type == VectorType.FLOAT:
            dtype = torch.float32
        elif self.config.vector_type == VectorType.FLOAT16:
            dtype = torch.float16
        elif self.config.vector_type == VectorType.BFLOAT16:
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown dtype: {self.config.vector_type}")
        return dtype

    @cached_property
    def _vector_numpy_dtype(self) -> type:
        dtype: type
        if self.config.vector_type == VectorType.FLOAT:
            dtype = np.float32
        elif self.config.vector_type == VectorType.FLOAT16:
            dtype = np.float16
        elif self.config.vector_type == VectorType.BFLOAT16:
            dtype = ml_dtypes.bfloat16
        else:
            raise ValueError(f"Unknown dtype: {self.config.vector_type}")
        return dtype

    def _convert_vector_to_array_for_insertion(self, vector: "Vector") -> "NumpyVector":
        dtype = self._vector_numpy_dtype

        if isinstance(vector, list):
            return np.array(vector, dtype=dtype)
        if isinstance(vector, np.ndarray):
            return vector.astype(dtype)
        if isinstance(vector, torch.Tensor):
            return (
                vector.cpu()
                .type(self._vector_torch_dtype)
                .view(dtype=torch.int8)
                .numpy()
                .view(dtype=dtype)
            )
        raise ValueError(f"Unknown vector type: {type(vector)}")
