from __future__ import annotations

from typing import TYPE_CHECKING

from contextualization.core.entities.config.context_summarizer import (
    LLMContextSummarizerConfig,
)
from contextualization.core.entities.config.publication_encoder import (
    LLMPublicationEncoderConfig,
)
from contextualization.core.entities.config.publication_scorer import (
    AggregatedPublicationScorerConfig,
    CoveragePublicationScorerConfig,
    RecencyPublicationScorerConfig,
)
from contextualization.core.entities.config.publication_vector_database import (
    InMemoryPublicationVectorDatabaseConfig,
    MilvusPublicationVectorDatabaseConfig,
)
from contextualization.core.entities.config.reference_database import (
    NdjsonReferenceDatabaseConfig,
    PostgresReferenceDatabaseConfig,
)
from contextualization.core.entities.config.reference_reader import (
    NdjsonReferenceReaderConfig,
    ParquetReferenceReaderConfig,
)
from contextualization.core.entities.config.retriever import (
    LookupRetrieverConfig,
    RetrieverConfig,
)
from contextualization.impl.context_summarizer import LLMContextSummarizer
from contextualization.impl.publication_encoder import LLMPublicationEncoder
from contextualization.impl.publication_scorer import CoveragePublicationScorer
from contextualization.impl.publication_scorer.aggregated import (
    AggregatedPublicationScorer,
)
from contextualization.impl.publication_scorer.recency import RecencyPublicationScorer
from contextualization.impl.publication_vector_database import (
    InMemoryPublicationVectorDatabase,
    MilvusPublicationVectorDatabase,
)
from contextualization.impl.reference_database import (
    NdjsonReferenceDatabase,
    PostgresReferenceDatabase,
)
from contextualization.impl.reference_reader import NdjsonReferenceReader, ParquetReferenceReader
from contextualization.impl.retriever import LookupRetriever

if TYPE_CHECKING:
    from collections.abc import Mapping

    from contextualization.core.contracts.context_summarizer import ContextSummarizer
    from contextualization.core.contracts.publication_encoder import PublicationEncoder
    from contextualization.core.contracts.publication_scorer import PublicationScorer
    from contextualization.core.contracts.publication_vector_database import (
        PublicationVectorDatabase,
    )
    from contextualization.core.contracts.reference_database import ReferenceDatabase
    from contextualization.core.contracts.reference_reader import ReferenceReader
    from contextualization.core.contracts.retriever import Retriever
    from contextualization.core.entities.config import (
        ContextSummarizerConfig,
        PublicationEncoderConfig,
        PublicationScorerConfig,
        PublicationVectorDatabaseConfig,
        ReferenceDatabaseConfig,
        ReferenceReaderConfig,
    )

REFERENCE_READER_CLASS_MAPPING: Mapping[type[ReferenceReaderConfig], type[ReferenceReader]] = {
    NdjsonReferenceReaderConfig: NdjsonReferenceReader,
    ParquetReferenceReaderConfig: ParquetReferenceReader,
}

REFERENCE_DATABASE_CLASS_MAPPING: Mapping[
    type[ReferenceDatabaseConfig], type[ReferenceDatabase]
] = {
    NdjsonReferenceDatabaseConfig: NdjsonReferenceDatabase,
    PostgresReferenceDatabaseConfig: PostgresReferenceDatabase,
}

PUBLICATION_VECTOR_DATABASE_CLASS_MAPPING: Mapping[
    type[PublicationVectorDatabaseConfig], type[PublicationVectorDatabase]
] = {
    MilvusPublicationVectorDatabaseConfig: MilvusPublicationVectorDatabase,
    InMemoryPublicationVectorDatabaseConfig: InMemoryPublicationVectorDatabase,
}

PUBLICATION_ENCODER_CLASS_MAPPING: Mapping[
    type[PublicationEncoderConfig], type[PublicationEncoder]
] = {
    LLMPublicationEncoderConfig: LLMPublicationEncoder,
}


PUBLICATION_SCORER_CLASS_MAPPING: Mapping[
    type[PublicationScorerConfig], type[PublicationScorer]
] = {
    CoveragePublicationScorerConfig: CoveragePublicationScorer,
    RecencyPublicationScorerConfig: RecencyPublicationScorer,
    AggregatedPublicationScorerConfig: AggregatedPublicationScorer,
}

RETRIEVER_CLASS_MAPPING: Mapping[type[RetrieverConfig], type[Retriever]] = {
    LookupRetrieverConfig: LookupRetriever
}

CONTEXT_SUMMARIZER_CLASS_MAPPING: Mapping[
    type[ContextSummarizerConfig], type[ContextSummarizer]
] = {
    LLMContextSummarizerConfig: LLMContextSummarizer,
}
