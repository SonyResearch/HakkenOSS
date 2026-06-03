from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from query_common.entities.query import Candidate

    from simple_query.kg.base import KnowledgeGraph
    from simple_query.kg.entities.configs import KnowledgeGraphConfigBase
    from simple_query.link_predictor.base import LinkPredictor
    from simple_query.link_predictor.entities.configs import LinkPredictorConfigBase
    from simple_query.query.entities.inputs import QueryInput


T = TypeVar("T")
KGConfigT = TypeVar("KGConfigT", bound="KnowledgeGraphConfigBase")
LinkPredictorConfigT = TypeVar("LinkPredictorConfigT", bound="LinkPredictorConfigBase")


class Querying(ABC, Generic[T]):
    """
    Base class for querying implementations.
    It is defined as a generic class, so that the implementation can be coupled with
    its corresponding config class for more comprehensive type annotations.
    """

    def __init__(
        self,
        config: T,
        kg: "KnowledgeGraph[KGConfigT]",
        link_predictor: "LinkPredictor[LinkPredictorConfigT]",
    ) -> None:
        self.config = config
        self.kg = kg
        self.link_predictor = link_predictor

    @abstractmethod
    def find_candidates(self, query_input: "QueryInput") -> list["Candidate"]:
        raise NotImplementedError
