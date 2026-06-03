from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING

import ml_dtypes
import numpy as np
import torch

from contextualization.core.contracts.publication_vector_database import (
    PublicationVectorDatabase,
)
from contextualization.core.entities.config.publication_vector_database import (
    InMemoryPublicationVectorDatabaseConfig,
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


class InMemoryPublicationVectorDatabase(
    PublicationVectorDatabase[InMemoryPublicationVectorDatabaseConfig]
):
    def __init__(self, config: InMemoryPublicationVectorDatabaseConfig):
        super().__init__(config)

        self.data: dict[PublicationId, dict[int, VectorWithMetadata]] = defaultdict(lambda: {})

    def exists(self, publication_id: "PublicationId", chunk_index: int | None = None) -> bool:
        if publication_id in self.data:
            if chunk_index is None:
                return bool(self.data[publication_id])
            return chunk_index in self.data[publication_id]
        return False

    def retrieve_statistics(self) -> VectorDatabaseStatistics:
        complete_publication_ids: list[PublicationId] = []
        chunk_existence: dict[PublicationId, set[int]] = defaultdict(lambda: set())

        for metadata in self.metadata_iterator():
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
        for publication_id in self.data:
            for vector_with_metadata in self.data[publication_id].values():
                metadata = vector_with_metadata.metadata
                if not include_text:
                    yield metadata.model_copy(update={"text": None})
                else:
                    yield metadata.model_copy()

    def get_by_publication_ids(
        self,
        publication_ids: "Sequence[PublicationId]",
        include_text: bool = False,
    ) -> "Sequence[VectorWithMetadata]":
        results: list[VectorWithMetadata] = []

        for pub_id in publication_ids:
            for vector_with_metadata in self.data[pub_id].values():
                update = None if include_text else {"text": None}
                metadata = vector_with_metadata.metadata.model_copy(update=update)
                results.append(vector_with_metadata.model_copy(update={"metadata": metadata}))

        return results

    def insert(self, data: "Sequence[VectorWithMetadata]") -> None:
        for vector_with_metadata in data:
            vector = self._convert_vector_to_array_for_insertion(vector_with_metadata.vector)
            pub_id = vector_with_metadata.metadata.publication_id
            chunk_index = vector_with_metadata.metadata.chunk_index
            self.data[pub_id][chunk_index] = vector_with_metadata.model_copy(
                update={"vector": vector}
            )

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
