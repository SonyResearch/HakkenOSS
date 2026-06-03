from pathlib import Path
from typing import Literal

import pytest
from dotenv import load_dotenv
from filtering.container import FilteringSettings
from filtering.impl.kg.neo4j_kg import Neo4jKnowledgeGraph as FilteringNeo4jKG
from filtering.impl.node_filtering import EntropyNodeFiltering
from query_common.parse.base import Parser
from query_common.parse.impl.lark import LarkParser

from complex_query.container import QueryingContainer, QueryingSettings
from complex_query.core.contracts import (
    KnowledgeGraph,
    LinkPredictor,
    ScoreLedger,
    Search,
)
from complex_query.core.entities.config.kg_ledger import HDF5KGLedgerConfig
from complex_query.impl.kg.neo4j_kg import Neo4jKG
from complex_query.impl.link_predictor import RandomLinkPredictor
from complex_query.impl.score_ledger.in_memory import InMemoryScoreLedger
from complex_query.impl.search.beam_search import QueryBeamSearch


@pytest.mark.parametrize("parser_type", ["lark"])
@pytest.mark.parametrize("search_config", [{"config_type": "beam"}])
@pytest.mark.parametrize("kg_config", [{"config_type": "networkx"}, {"config_type": "neo4j"}])
@pytest.mark.parametrize("kg_ledger_config", [{"config_type": "hdf5"}])
@pytest.mark.parametrize("use_kg_ledger", [True, False])
@pytest.mark.parametrize("score_ledger_config", [{"config_type": "in_memory"}])
@pytest.mark.parametrize(
    "link_predictor_config", [{"config_type": "random"}, {"config_type": "api_based"}]
)
def test_container_init(  # noqa: PLR0913
    parser_type: Literal["lark"],
    search_config: dict[str, str],
    kg_config: dict[str, str] | dict[str, str | bool],
    kg_ledger_config: dict[str, str],
    use_kg_ledger: bool,
    score_ledger_config: dict[str, str],
    link_predictor_config: dict[str, str],
    tmp_path: Path,
):
    filtering_container_config = FilteringSettings(
        kg_config={"config_type": "neo4j"}  # type: ignore
    )

    if isinstance(kg_ledger_config, HDF5KGLedgerConfig):
        kg_ledger_config.file_path = tmp_path

    config = QueryingSettings(
        parser_type=parser_type,  # type: ignore
        search_config=search_config,  # type: ignore
        kg_config=kg_config,  # type: ignore
        kg_ledger_config=kg_ledger_config,  # type: ignore
        score_ledger_config=score_ledger_config,  # type: ignore
        link_predictor_config=link_predictor_config,  # type: ignore
        filtering_container_config=filtering_container_config,  # type: ignore
        use_kg_ledger=use_kg_ledger,
        use_filtering=True,
    )

    container = QueryingContainer()
    container.config.from_pydantic(config)

    container.wire(modules=[__name__], packages=["complex_query"])

    assert isinstance(container.parser(), Parser)
    assert isinstance(container.kg(), KnowledgeGraph)
    assert isinstance(container.score_ledger(), ScoreLedger)
    assert isinstance(container.link_predictor(), LinkPredictor)
    assert isinstance(container.search(), Search)


def test_container_init_from_env():
    load_dotenv(Path(__file__).parent / "test_container.env", override=True)

    config = QueryingSettings()
    container = QueryingContainer()
    container.config.from_pydantic(config)

    container.wire(modules=[__name__], packages=["complex_query"])

    parser = container.parser()
    kg = container.kg()
    score_ledger = container.score_ledger()
    link_predictor = container.link_predictor()
    search = container.search()
    node_filtering = container.filtering_container.node_filtering()

    assert isinstance(parser, LarkParser)
    assert isinstance(kg, Neo4jKG)
    assert isinstance(score_ledger, InMemoryScoreLedger)
    assert isinstance(link_predictor, RandomLinkPredictor)
    assert isinstance(search, QueryBeamSearch)
    assert isinstance(node_filtering, EntropyNodeFiltering)
    assert isinstance(node_filtering.kg, FilteringNeo4jKG)
