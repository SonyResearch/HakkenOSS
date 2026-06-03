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

from simple_query.api.errors import InitializationError
from simple_query.kg.entities.configs import KnowledgeGraphConfig
from simple_query.kg.values.class_mapping import CLASS_MAPPING as KG_CLASS_MAPPING
from simple_query.link_predictor.entities.configs import LinkPredictorConfig
from simple_query.link_predictor.values.class_mapping import (
    CLASS_MAPPING as LINK_PREDICTOR_CLASS_MAPPING,
)
from simple_query.query.entities.configs import QueryingConfig
from simple_query.query.values.class_mapping import CLASS_MAPPING as QUERYING_CLASS_MAPPING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from simple_query.kg.base import KnowledgeGraph
    from simple_query.link_predictor.base import LinkPredictor
    from simple_query.query.base import Querying

ConfigT = TypeVar("ConfigT")
ModelT = TypeVar("ModelT")
P = ParamSpec("P")


def _initialize_model_factory(
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
def initialize_kg(
    config: "KnowledgeGraphConfig | dict[str, Any]" = Provide["config.kg_config"],
) -> "KnowledgeGraph":
    return _initialize_model_factory(KnowledgeGraphConfig, KG_CLASS_MAPPING)(config)


@inject
def initialize_link_predictor(
    config: LinkPredictorConfig | dict[str, Any] = Provide["config.link_predictor_config"],
) -> "LinkPredictor":
    return _initialize_model_factory(LinkPredictorConfig, LINK_PREDICTOR_CLASS_MAPPING)(config)


@inject
def initialize_querying(
    config: QueryingConfig | dict[str, Any] = Provide["config.querying_config"],
    kg: "KnowledgeGraph" = Provide["kg"],
    link_predictor: "LinkPredictor" = Provide["link_predictor"],
) -> "Querying":
    return _initialize_model_factory(QueryingConfig, QUERYING_CLASS_MAPPING)(
        config, kg=kg, link_predictor=link_predictor
    )
