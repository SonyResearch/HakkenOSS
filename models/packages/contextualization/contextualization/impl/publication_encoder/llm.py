import hashlib
import itertools
import logging
from typing import TYPE_CHECKING, cast

import torch
import torch.distributed as dist
from transformers import AutoModel, AutoTokenizer, BatchEncoding, PretrainedConfig

from contextualization.core.contracts.publication_encoder import PublicationEncoder
from contextualization.core.entities.config.publication_encoder import (
    LLMPublicationEncoderConfig,
)
from contextualization.core.entities.vector_database import (
    Metadata,
    VectorDatabaseStatistics,
    VectorWithMetadata,
)
from contextualization.core.values.errors import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from contextualization.core.contracts.publication_vector_database import (
        PublicationVectorDatabase,
    )
    from contextualization.core.contracts.reference_reader import ReferenceReader
    from contextualization.core.entities.publication import Publication, PublicationId

logger = logging.getLogger(__name__)


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_state[:, -1]

    sequence_lengths = attention_mask.sum(dim=1)
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths - 1
    ]


def avg_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden_state = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    sequence_lengths = attention_mask.sum(dim=1)
    return (last_hidden_state / sequence_lengths[:, None, None]).sum(dim=1)


class LLMPublicationEncoder(PublicationEncoder[LLMPublicationEncoderConfig]):
    def __init__(self, config: LLMPublicationEncoderConfig) -> None:
        super().__init__(config)

        self.model: torch.nn.Module
        self.model_type: str
        self.vector_dim: int

        self.model = AutoModel.from_pretrained(
            config.hf_model_name_or_path,
            trust_remote_code=True,
            **self.config.hf_model_loading_kwargs,
        )
        hf_config = cast("PretrainedConfig", self.model.config)
        self.model_type = hf_config.model_type
        self.vector_dim = hf_config.hidden_size

        self.model.to(self.config.device)
        if dist.is_initialized():
            if self.config.device is not None:
                raise ValueError(
                    "In a distributed environment, `device` will be automatically set to "
                    "local rank, thus should NOT be given."
                )
            self.model.to(dist.get_rank())

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.hf_model_name_or_path, **self.config.tokenizer_kwargs
        )

        self.complete_publication_ids: set[PublicationId] | None = None

        # Precomputate values required for tokenization
        self._add_prefix_space = self._requires_prefix_space()

        self._title_prefix_ids: list[int]
        self._joiner_ids: list[int]
        self._abstract_prefix_ids: list[int]
        self._precompute_ids()

        self._max_content_length: int
        self._title_max_length: int
        self._abstract_max_length: int
        self._compute_max_lengths()

    def _requires_prefix_space(self) -> bool:
        """Detect whether it requires prepend space for continuating text."""
        ids_without_space = self.tokenizer.encode("a", add_special_tokens=False)
        ids_with_space = self.tokenizer.encode(" a", add_special_tokens=False)
        return ids_without_space[0] != ids_with_space[0]

    def _precompute_ids(self) -> None:
        title_prefix = self.config.title_prefix
        joiner = self.config.joiner
        abstract_prefix = self.config.abstract_prefix
        if self._add_prefix_space:
            joiner = " " + joiner
            abstract_prefix = " " + abstract_prefix

        self._title_prefix_ids = self.tokenizer.encode(title_prefix, add_special_tokens=False)
        self._joiner_ids = self.tokenizer.encode(joiner, add_special_tokens=False)
        self._abstract_prefix_ids = self.tokenizer.encode(abstract_prefix, add_special_tokens=False)

    def _compute_max_lengths(self) -> None:
        max_length = self.config.max_length
        if max_length is None:
            max_length = int(1e12)
        num_special_tokens_to_add = self.tokenizer.num_special_tokens_to_add()

        self._max_content_length = (
            max_length
            - len(self._title_prefix_ids)
            - len(self._abstract_prefix_ids)
            - num_special_tokens_to_add
        )
        self._title_max_length = int(self.config.title_max_ratio * self._max_content_length)
        self._abstract_max_length = self._max_content_length - self._title_max_length

    def encode_and_store_to_db(
        self,
        reference_reader: "ReferenceReader",
        publication_vector_database: "PublicationVectorDatabase",
        skip_existing: bool = True,
    ) -> None:
        device = None
        if dist.is_initialized():
            device = torch.device(dist.get_rank())
        if self.config.device is not None:
            device = torch.device(self.config.device)

        for inputs_batch, metadata_batch in self.batch_iterator(
            reference_reader=reference_reader,
            batch_size=self.config.batch_size,
            publication_vector_database=publication_vector_database,
            skip_existing=skip_existing,
        ):
            inputs_batch_in_device = inputs_batch
            if device is not None:
                inputs_batch_in_device = inputs_batch.to(device)
            input_ids = cast("torch.Tensor", inputs_batch_in_device["input_ids"])
            attention_mask = cast("torch.Tensor", inputs_batch_in_device["attention_mask"])
            embeddings = (
                self._encode(input_ids=input_ids, attention_mask=attention_mask).cpu().unbind(dim=0)
            )

            vector_with_metadata_list = [
                VectorWithMetadata(vector=vector, metadata=metadata)
                for vector, metadata in zip(embeddings, metadata_batch, strict=True)
            ]
            publication_vector_database.insert(data=vector_with_metadata_list)

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: `torch.LongTensor` having indices of tokens, of shape `(batch_size, length)`.
            attention_mask: `torch.BoolTensor` of shape `(batch_size, length)`.
        """
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state

            if self.config.pooling_method == "last":
                embeddings = last_token_pool(
                    last_hidden_state=last_hidden_state, attention_mask=attention_mask
                )
            elif self.config.pooling_method == "avg":
                embeddings = avg_pool(
                    last_hidden_state=last_hidden_state, attention_mask=attention_mask
                )
            else:
                raise ConfigurationError(f"unknown pooling_method: {self.config.pooling_method}")

        return embeddings

    @staticmethod
    def _check_inclusion_condition(
        skip_existing: bool,
        publication_vector_database: "PublicationVectorDatabase | None",
        publication_vector_database_statistics: VectorDatabaseStatistics | None,
        publication_id: "PublicationId",
        chunk_index: int,
    ) -> bool:
        if not skip_existing:
            return True

        publication_vector_database = cast("PublicationVectorDatabase", publication_vector_database)

        if publication_vector_database_statistics is not None:
            if publication_id in publication_vector_database_statistics.finished:
                return False
            if publication_id not in publication_vector_database_statistics.partial:
                return True

        return not publication_vector_database.exists(
            publication_id=publication_id, chunk_index=chunk_index
        )

    def _get_db_statistics(
        self, publication_vector_database: "PublicationVectorDatabase"
    ) -> VectorDatabaseStatistics:
        is_distributed = dist.is_initialized()

        if is_distributed:
            rank = dist.get_rank()

            object_list: list[VectorDatabaseStatistics]
            if rank == 0:
                logger.info(f"Rank {rank}: Retrieving DB statistics")
                db_statistics = publication_vector_database.retrieve_statistics()
                object_list = [db_statistics]
            else:
                object_list = [None]  # type: ignore

            if rank == 0:
                logger.info(f"Rank {rank}: Broadcasting the statistics to other processes")
            dist.broadcast_object_list(object_list, src=0)
            db_statistics = object_list[0]
        else:
            db_statistics = publication_vector_database.retrieve_statistics()

        return db_statistics

    def _encode_publication(
        self,
        publication: "Publication",
        skip_existing: bool,
        publication_vector_database: "PublicationVectorDatabase | None",
        publication_vector_database_statistics: VectorDatabaseStatistics | None,
    ) -> tuple[list[list[int]], list[Metadata]]:
        title = publication.title or ""
        abstract = publication.abstract or ""
        overlap = self.config.overlap

        if self._add_prefix_space:
            title = " " + title
            abstract = " " + abstract

        title_ids = self.tokenizer.encode(title, add_special_tokens=False)
        abstract_ids = self.tokenizer.encode(abstract, add_special_tokens=False)

        if len(title_ids) + len(abstract_ids) <= self._max_content_length:
            title_ids_chunks = [title_ids]
            abstract_ids_chunks = [abstract_ids]
        else:
            title_ids_chunks = [
                title_ids[i : i + self._title_max_length]
                for i in range(0, len(title_ids) - overlap, self._title_max_length - overlap)
            ]
            abstract_ids_chunks = [
                abstract_ids[i : i + self._abstract_max_length]
                for i in range(0, len(abstract_ids) - overlap, self._abstract_max_length - overlap)
            ]

        input_ids_list: list[list[int]] = []
        metadata_list: list[Metadata] = []

        num_chunks = len(title_ids_chunks) * len(abstract_ids_chunks)

        for i, (title_ids_chunk, abstract_ids_chunk) in enumerate(
            itertools.product(title_ids_chunks, abstract_ids_chunks)
        ):
            if self._check_inclusion_condition(
                skip_existing=skip_existing,
                publication_vector_database=publication_vector_database,
                publication_vector_database_statistics=publication_vector_database_statistics,
                publication_id=publication.publication_id,
                chunk_index=i + 1,
            ):
                input_ids_chunk = (
                    self._title_prefix_ids
                    + title_ids_chunk
                    + self._joiner_ids
                    + self._abstract_prefix_ids
                    + abstract_ids_chunk
                )
                input_ids_chunk = self.tokenizer.build_inputs_with_special_tokens(input_ids_chunk)
                metadata = Metadata(
                    publication_id=publication.publication_id,
                    chunk_index=i + 1,
                    num_chunks=num_chunks,
                    text=" ".join(self.tokenizer.decode(input_ids_chunk)),
                )

                input_ids_list.append(input_ids_chunk)
                metadata_list.append(metadata)

        return input_ids_list, metadata_list

    def batch_iterator(
        self,
        reference_reader: "ReferenceReader",
        batch_size: int,
        publication_vector_database: "PublicationVectorDatabase | None" = None,
        skip_existing: bool = True,
    ) -> "Iterator[tuple[BatchEncoding, list[Metadata]]]":
        """
        Iterates data into batches acceptable from the LLM,
        i.e. batch of tokenized input IDs and attention masks.

        For a long text whose length exceeds the maximum length of the LLM,
        it splits the text into chunks, while ensuring the title information is available
        in every chunk.

        It can optionally skip encoding of vectors that already exist in a vector database.
        In a distributed setup, it distributes data into multiple processes based on
        the MD5 hash of each publication ID, to efficiently distributed data while
        keeping the capability of skipping existing data.

        Args:
            reference_reader: `ReferenceReader` for the data to be iterated.
            batch_size: Batch size.
            publication_vector_database:
                `PublicationVectorDatabase` object to which encoded vectors will be stored.
            skip_existing: Whether to skip vectors already exist in `publication_vector_database`.
        """
        if skip_existing and publication_vector_database is None:
            raise ValueError(
                "`publication_vector_database` cannot be `None` when `skip_existing` is `True`."
            )

        # In a distributed setup where `skip_existing` is `True`, simply distributing data
        # into multiple processes may result in incorrect allocation.
        # Instead, we will distribute data by taking a MD5 hash of publication ID,
        # which is proven to generate uniform values in a deterministic way.
        is_distributed = dist.is_initialized()
        rank = 0
        world_size = 1
        if is_distributed:
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        # Prefetch publication IDs all of whose chunk vectors exist in the database
        db_statistics: VectorDatabaseStatistics | None = None
        if skip_existing:
            publication_vector_database = cast(
                "PublicationVectorDatabase", publication_vector_database
            )
            db_statistics = self._get_db_statistics(
                publication_vector_database=publication_vector_database
            )

        input_ids_queue: list[list[int]] = []
        metadata_queue: list[Metadata] = []

        for publication in reference_reader.iter_publications():
            while len(input_ids_queue) >= batch_size:
                padded_inputs = self.tokenizer.pad(
                    {"input_ids": input_ids_queue[:batch_size]}, return_tensors="pt"
                )
                yield padded_inputs, metadata_queue[:batch_size]
                input_ids_queue = input_ids_queue[batch_size:]
                metadata_queue = metadata_queue[batch_size:]

            if db_statistics is not None and publication.publication_id in db_statistics.finished:
                continue

            if is_distributed:
                publication_ids_hash = hashlib.md5(
                    publication.publication_id.encode("utf-8")
                ).hexdigest()
                bucket_index = int(publication_ids_hash[:8], 16) % world_size
                if rank != bucket_index:
                    continue

            publication_input_ids_list, publication_metadata_list = self._encode_publication(
                publication=publication,
                skip_existing=skip_existing,
                publication_vector_database=publication_vector_database,
                publication_vector_database_statistics=db_statistics,
            )
            input_ids_queue.extend(publication_input_ids_list)
            metadata_queue.extend(publication_metadata_list)

        while len(input_ids_queue) > 0:
            padded_inputs = self.tokenizer.pad(
                {"input_ids": input_ids_queue[:batch_size]}, return_tensors="pt"
            )
            yield padded_inputs, metadata_queue[:batch_size]
            input_ids_queue = input_ids_queue[batch_size:]
            metadata_queue = metadata_queue[batch_size:]
