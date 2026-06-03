from __future__ import annotations

from typing import TYPE_CHECKING

from complex_query.core.entities.config.kg import (
    KGConfigBase,
    Neo4jKGConfig,
    NetworkxKGConfig,
)
from complex_query.core.entities.config.kg_ledger import (
    HDF5KGLedgerConfig,
    InMemoryKGLedgerConfig,
    KGLedgerConfigBase,
    SqliteKGLedgerConfig,
)
from complex_query.core.entities.config.link_predictor import (
    ApiBasedLinkPredictorConfig,
    LinkPredictorConfigBase,
    RandomLinkPredictorConfig,
)
from complex_query.core.entities.config.score_aggregator import (
    MinimumScoreAggregatorConfig,
    ProductScoreAggregatorConfig,
    ScoreAggregatorConfigBase,
)
from complex_query.core.entities.config.score_ledger import (
    InMemoryScoreLedgerConfig,
    ScoreLedgerConfigBase,
)
from complex_query.core.entities.config.search import BeamSearchConfig, SearchConfigBase
from complex_query.impl.kg.neo4j_kg import Neo4jKG
from complex_query.impl.kg.networkx_kg import NetworkxKG
from complex_query.impl.kg_ledger.hdf5 import HDF5KnowledgeGraphLedger
from complex_query.impl.kg_ledger.in_memory import InMemoryKnowledgeGraphLedger
from complex_query.impl.kg_ledger.sqlite import SqliteKnowledgeGraphLedger
from complex_query.impl.link_predictor import (
    ApiBasedLinkPredictor,
    RandomLinkPredictor,
)
from complex_query.impl.score_aggregator.minimum import MinimumScoreAggregator
from complex_query.impl.score_aggregator.product import ProductScoreAggregator
from complex_query.impl.score_ledger.in_memory import InMemoryScoreLedger
from complex_query.impl.search.beam_search import QueryBeamSearch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from complex_query.core.contracts import (
        KnowledgeGraph,
        KnowledgeGraphLedger,
        LinkPredictor,
        ScoreAggregator,
        ScoreLedger,
        Search,
    )


SEARCH_CLASS_MAPPING: Mapping[type[SearchConfigBase], type[Search]] = {
    BeamSearchConfig: QueryBeamSearch,
}

KG_CLASS_MAPPING: Mapping[type[KGConfigBase], type[KnowledgeGraph]] = {
    NetworkxKGConfig: NetworkxKG,
    Neo4jKGConfig: Neo4jKG,
}

KG_LEDGER_CLASS_MAPPING: Mapping[type[KGLedgerConfigBase], type[KnowledgeGraphLedger]] = {
    HDF5KGLedgerConfig: HDF5KnowledgeGraphLedger,
    InMemoryKGLedgerConfig: InMemoryKnowledgeGraphLedger,
    SqliteKGLedgerConfig: SqliteKnowledgeGraphLedger,
}

SCORE_LEDGER_CLASS_MAPPING: Mapping[type[ScoreLedgerConfigBase], type[ScoreLedger]] = {
    InMemoryScoreLedgerConfig: InMemoryScoreLedger,
}

SCORE_AGGREGATOR_CLASS_MAPPING: Mapping[type[ScoreAggregatorConfigBase], type[ScoreAggregator]] = {
    ProductScoreAggregatorConfig: ProductScoreAggregator,
    MinimumScoreAggregatorConfig: MinimumScoreAggregator,
}

LINK_PREDICTOR_CLASS_MAPPING: Mapping[type[LinkPredictorConfigBase], type[LinkPredictor]] = {
    RandomLinkPredictorConfig: RandomLinkPredictor,
    ApiBasedLinkPredictorConfig: ApiBasedLinkPredictor,
}
