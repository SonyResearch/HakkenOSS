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

from complex_query.core.entities.config.kg import KGConfig
from complex_query.core.entities.config.kg_ledger import KGLedgerConfig
from complex_query.core.entities.config.link_predictor import LinkPredictorConfig
from complex_query.core.entities.config.score_aggregator import ScoreAggregatorConfig
from complex_query.core.entities.config.score_ledger import ScoreLedgerConfig
from complex_query.core.entities.config.search import SearchConfig
from complex_query.core.values.class_mapping import (
    KG_CLASS_MAPPING,
    KG_LEDGER_CLASS_MAPPING,
    LINK_PREDICTOR_CLASS_MAPPING,
    SCORE_AGGREGATOR_CLASS_MAPPING,
    SCORE_LEDGER_CLASS_MAPPING,
    SEARCH_CLASS_MAPPING,
)
from complex_query.core.values.errors import InitializationError
from complex_query.impl.kg.cached_kg import CachedKnowledgeGraph

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from filtering.core.contracts import NodeFiltering

    from complex_query.core.contracts import (
        KnowledgeGraph,
        KnowledgeGraphLedger,
        LinkPredictor,
        ScoreLedger,
        Search,
    )
    from complex_query.core.contracts.score_aggregator import ScoreAggregator


ConfigT = TypeVar("ConfigT")
ModelT = TypeVar("ModelT")
P = ParamSpec("P")


def _initializer_factory(
    config_class: Any, class_mapping: "Mapping[type[ConfigT], type[ModelT]]"
) -> "Callable[Concatenate[ConfigT | dict[str, Any], P], ModelT]":
    """
    A factory function that returns the corresponding initializer function
    based on a config class value (which is a discriminated union, i.e. `Annotated[Union[...]]`)
    and a mapping from a config class to its model class.
    """

    config_type_origin = get_origin(config_class)
    if config_type_origin is not Annotated:
        # The origin of `config_class` should be `Annotated`, since we are using `TypeAdapter`.
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
                f"and could not be converted into `{config_type_base!r}` "
                f"(got: {type(config)!r})"
            )

        object_class = class_mapping[type(config)]
        return object_class(config, *args, **kwargs)  # type: ignore

    return initialize_object


@inject
def initialize_kg_ledger(
    config: "KGLedgerConfig | dict[str, Any]" = Provide["config.kg_ledger_config"],
) -> "KnowledgeGraphLedger":
    return _initializer_factory(KGLedgerConfig, KG_LEDGER_CLASS_MAPPING)(config)


@inject
def initialize_score_ledger(
    config: ScoreLedgerConfig | dict[str, Any] = Provide["config.score_ledger_config"],
) -> "ScoreLedger":
    return _initializer_factory(ScoreLedgerConfig, SCORE_LEDGER_CLASS_MAPPING)(config)


@inject
def initialize_kg(
    config: KGConfig | dict[str, Any] = Provide["config.kg_config"],
    use_kg_ledger: bool = Provide["config.use_kg_ledger"],
    kg_ledger_config: KGLedgerConfig | dict[str, Any] = Provide["config.kg_ledger_config"],
) -> "KnowledgeGraph":
    base_kg: KnowledgeGraph
    kg: KnowledgeGraph

    base_kg = _initializer_factory(KGConfig, class_mapping=KG_CLASS_MAPPING)(config)

    if use_kg_ledger:
        if not kg_ledger_config:
            raise ValueError("kg_ledger_type must be given when use_kg_ledger is True")

        kg_ledger = initialize_kg_ledger()
        kg = CachedKnowledgeGraph(base_kg=base_kg, ledger=kg_ledger)
    else:
        kg = base_kg

    return kg


@inject
def initialize_search(  # noqa: PLR0913
    config: SearchConfig | dict[str, Any] = Provide["config.search_config"],
    kg: "KnowledgeGraph" = Provide["kg"],
    score_ledger: "ScoreLedger" = Provide["score_ledger"],
    score_aggregator: "ScoreAggregator" = Provide["score_aggregator"],
    link_predictor: "LinkPredictor" = Provide["link_predictor"],
    node_filtering: "NodeFiltering" = Provide["node_filtering"],
) -> "Search":
    return _initializer_factory(SearchConfig, SEARCH_CLASS_MAPPING)(
        config,
        kg=kg,
        ledger=score_ledger,
        link_predictor=link_predictor,
        node_filtering=node_filtering,
        score_aggregator=score_aggregator,
    )


@inject
def initialize_score_aggregator(
    config: ScoreAggregatorConfig | dict[str, Any] = Provide["config.score_aggregator_config"],
) -> "ScoreAggregator":
    return _initializer_factory(ScoreAggregatorConfig, SCORE_AGGREGATOR_CLASS_MAPPING)(config)


@inject
def initialize_link_predictor(
    config: LinkPredictorConfig | dict[str, Any] = Provide["config.link_predictor_config"],
) -> "LinkPredictor":
    return _initializer_factory(LinkPredictorConfig, LINK_PREDICTOR_CLASS_MAPPING)(config)
