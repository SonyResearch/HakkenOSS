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

from filtering.core.entities.config.knowledge_graph import KnowledgeGraphConfig
from filtering.core.entities.config.node_filtering import NodeFilteringConfig
from filtering.core.entities.config.triple_filtering import TripleFilteringConfig
from filtering.core.values.class_mapping import (
    KG_CLASS_MAPPING,
    NODE_FILTERING_CLASS_MAPPING,
    TRIPLE_FILTERING_CLASS_MAPPING,
)
from filtering.core.values.errors import InitializationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from filtering.core.contracts import KnowledgeGraph, NodeFiltering, TripleFiltering

ConfigT = TypeVar("ConfigT")
ModelT = TypeVar("ModelT")
P = ParamSpec("P")


def _initializer_factory(
    config_class, class_mapping: "Mapping[type[ConfigT], type[ModelT]]"
) -> "Callable[Concatenate[ConfigT | dict[str, Any], P], ModelT]":
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
                f"and could not be converted into `{config_type_base!r}` "
                f"(got: {type(config)!r})"
            )

        object_class = class_mapping[type(config)]
        return object_class(config, *args, **kwargs)  # type: ignore

    return initialize_object


@inject
def initialize_kg(
    config: KnowledgeGraphConfig | dict[str, Any] = Provide["config.kg_config"],
) -> "KnowledgeGraph":
    return _initializer_factory(KnowledgeGraphConfig, KG_CLASS_MAPPING)(config)


@inject
def initialize_node_filtering(
    config: NodeFilteringConfig | dict[str, Any] = Provide["config.node_filtering_config"],
    kg: "KnowledgeGraph | None" = Provide["kg"],
) -> "NodeFiltering":
    return _initializer_factory(NodeFilteringConfig, NODE_FILTERING_CLASS_MAPPING)(config, kg=kg)


@inject
def initialize_triple_filtering(
    config: TripleFilteringConfig | dict[str, Any] = Provide["config.triple_filtering_config"],
    kg: "KnowledgeGraph | None" = Provide["kg"],
) -> "TripleFiltering":
    return _initializer_factory(TripleFilteringConfig, TRIPLE_FILTERING_CLASS_MAPPING)(
        config, kg=kg
    )
