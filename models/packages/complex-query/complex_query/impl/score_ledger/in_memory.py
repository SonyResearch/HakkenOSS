from typing import TYPE_CHECKING

from complex_query.core.contracts.score_ledger import ScoreLedger
from complex_query.core.entities.config.score_ledger import InMemoryScoreLedgerConfig

if TYPE_CHECKING:
    from query_common.entities.kg.identifier import ConceptIdentifier, RelationIdentifier
    from query_common.entities.kg.triple import Triple


class InMemoryScoreLedger(ScoreLedger[InMemoryScoreLedgerConfig]):
    def __init__(self, config: InMemoryScoreLedgerConfig) -> None:
        super().__init__(config)

        self.scores: dict[
            tuple[ConceptIdentifier, RelationIdentifier, ConceptIdentifier], float
        ] = {}

    def save_link_score(self, triple: "Triple", score: float) -> None:
        self.scores[
            (triple.subject_identifier, triple.relation_identifier, triple.object_identifier)
        ] = score

    def retrieve_link_score(self, triple: "Triple") -> float:
        try:
            return self.scores[
                (triple.subject_identifier, triple.relation_identifier, triple.object_identifier)
            ]
        except KeyError as e:
            raise KeyError(f"Triple {triple} not found in ScoreLedger") from e
