from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from filtering.core.contracts import NodeFiltering
    from query_common.entities.grounded_query import GroundedQuery
    from query_common.entities.query import Candidate

    from complex_query.core.contracts.kg import KnowledgeGraph
    from complex_query.core.contracts.link_predictor import LinkPredictor
    from complex_query.core.contracts.score_aggregator import ScoreAggregator
    from complex_query.core.contracts.score_ledger import ScoreLedger


T = TypeVar("T")


class Search(ABC, Generic[T]):
    def __init__(  # noqa: PLR0913
        self,
        config: T,
        kg: "KnowledgeGraph",
        ledger: "ScoreLedger",
        link_predictor: "LinkPredictor",
        node_filtering: "NodeFiltering",
        score_aggregator: "ScoreAggregator",
    ) -> None:
        self.config = config
        self.kg = kg
        self.ledger = ledger
        self.link_predictor = link_predictor
        self.node_filtering = node_filtering
        self.score_aggregator = score_aggregator

    @abstractmethod
    def find_candidates(self, query: "GroundedQuery", n_candidates: int) -> list["Candidate"]:
        pass
