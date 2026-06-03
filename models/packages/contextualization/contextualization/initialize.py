from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Concatenate,
    ParamSpec,
    TypeVar,
    get_args,
    get_origin,
)

from dependency_injector.wiring import Provide, inject
from pydantic import TypeAdapter

from contextualization.core.contracts.context_summarizer import (
    ContextSummarizer,
    ContextSummarizerToken,
)
from contextualization.core.contracts.publication_scorer import (
    PublicationScorer,
    PublicationScorerToken,
)
from contextualization.core.contracts.reference_database import (
    ReferenceDatabase,
    ReferenceDatabaseToken,
)
from contextualization.core.entities.config import (
    ContextSummarizerConfig,
    PublicationEncoderConfig,
    PublicationScorerConfig,
    PublicationVectorDatabaseConfig,
    ReferenceDatabaseConfig,
    ReferenceReaderConfig,
)
from contextualization.core.entities.config.retriever import RetrieverConfig
from contextualization.core.values.class_mapping import (
    CONTEXT_SUMMARIZER_CLASS_MAPPING,
    PUBLICATION_ENCODER_CLASS_MAPPING,
    PUBLICATION_SCORER_CLASS_MAPPING,
    PUBLICATION_VECTOR_DATABASE_CLASS_MAPPING,
    REFERENCE_DATABASE_CLASS_MAPPING,
    REFERENCE_READER_CLASS_MAPPING,
    RETRIEVER_CLASS_MAPPING,
)
from contextualization.core.values.errors import InitializationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from contextualization.core.contracts.publication_encoder import PublicationEncoder
    from contextualization.core.contracts.publication_vector_database import (
        PublicationVectorDatabase,
    )
    from contextualization.core.contracts.reference_reader import ReferenceReader
    from contextualization.core.contracts.retriever import Retriever

ConfigT = TypeVar("ConfigT")
ModelT = TypeVar("ModelT")
P = ParamSpec("P")


def _initialize_model_factory(
    config_class, class_mapping: "Mapping[type[ConfigT], type[ModelT]]"
) -> "Callable[Concatenate[ConfigT | dict[str, Any], P], ModelT]":
    """
    A factory function that returns the initializer function based on a type value.
    """

    config_type_origin = get_origin(config_class)
    if config_type_origin is not Annotated:
        raise InitializationError(
            f"`config_class` should be an annotated type, but got {config_class}"
        )
    config_type_args = get_args(config_class)
    config_type_base = config_type_args[0]

    def initialize_object(config: ConfigT | dict[str, Any], *args, **kwargs) -> ModelT:
        if isinstance(config, dict):
            config_ta: TypeAdapter[ConfigT] = TypeAdapter(config_class)
            config = config_ta.validate_python(config)

        if not isinstance(config, config_type_base):
            raise InitializationError(
                f"`config` is not in `{config_type_base!r}` "
                f"and could not be ocnverted into `{config_type_base!r}` "
                f"(got: {type(config)!r})"
            )

        object_class = class_mapping[type(config)]
        return object_class(config, *args, **kwargs)  # type: ignore

    return initialize_object


@inject
def initialize_reference_reader(
    config: "ReferenceReaderConfig | dict[str, Any]" = Provide["config.reference_reader_type"],
) -> "ReferenceReader":
    return _initialize_model_factory(ReferenceReaderConfig, REFERENCE_READER_CLASS_MAPPING)(config)


@inject
def initialize_reference_database(
    config: "ReferenceDatabaseConfig | dict[str, Any]" = Provide["config.reference_database_type"],
) -> ReferenceDatabase:
    return _initialize_model_factory(ReferenceDatabaseConfig, REFERENCE_DATABASE_CLASS_MAPPING)(
        config
    )


@inject
def initialize_publication_vector_database(
    config: "PublicationVectorDatabaseConfig | dict[str, Any]" = Provide[
        "config.publication_vector_database_config"
    ],
) -> "PublicationVectorDatabase":
    return _initialize_model_factory(
        PublicationVectorDatabaseConfig, PUBLICATION_VECTOR_DATABASE_CLASS_MAPPING
    )(config)


@inject
def initialize_publication_encoder(
    config: "PublicationEncoderConfig | dict[str, Any]" = Provide[
        "config.publication_encoder_config"
    ],
) -> "PublicationEncoder":
    return _initialize_model_factory(PublicationEncoderConfig, PUBLICATION_ENCODER_CLASS_MAPPING)(
        config
    )


@inject
def initialize_publication_scorer(
    config: "PublicationScorerConfig | dict[str, Any]" = Provide[
        "config.publication_scorer_config"
    ],
    reference_database: ReferenceDatabase = Provide[ReferenceDatabaseToken],
) -> PublicationScorer:
    return _initialize_model_factory(PublicationScorerConfig, PUBLICATION_SCORER_CLASS_MAPPING)(
        config, reference_database=reference_database
    )


@inject
def initialize_retriever(
    config: RetrieverConfig | dict[str, Any] = Provide["config.retriever_config"],
    reference_database: ReferenceDatabase = Provide[ReferenceDatabaseToken],
    publication_scorer: PublicationScorer = Provide[PublicationScorerToken],
    context_summarizer: ContextSummarizer | None = Provide[ContextSummarizerToken],
) -> "Retriever":
    return _initialize_model_factory(RetrieverConfig, RETRIEVER_CLASS_MAPPING)(
        config,
        reference_database=reference_database,
        publication_scorer=publication_scorer,
        context_summarizer=context_summarizer,
    )


@inject
def initialize_context_summarizer(
    config: "ContextSummarizerConfig | dict[str, Any] | None" = Provide[
        "config.context_summarizer_config"
    ],
) -> ContextSummarizer | None:
    if not config:
        return None

    return _initialize_model_factory(ContextSummarizerConfig, CONTEXT_SUMMARIZER_CLASS_MAPPING)(
        config
    )
