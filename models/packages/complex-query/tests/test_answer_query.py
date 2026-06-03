from query_common.entities.kg.concept import Concept
from query_common.entities.kg.triple import Triple
from query_common.entities.query import QueryRequest, RequestVariable
from query_common.values.types import ParserType

from complex_query.container import QueryingContainer, QueryingSettings
from complex_query.core.actions import answer_query
from complex_query.core.entities.config.kg import NetworkxKGConfig
from complex_query.core.entities.config.link_predictor import RandomLinkPredictorConfig
from complex_query.core.entities.config.score_aggregator import ProductScoreAggregatorConfig
from complex_query.core.entities.config.search import BeamSearchConfig


def test_answer_query():
    container = QueryingContainer()
    config = QueryingSettings(
        parser_type=ParserType.LARK,
        search_config=BeamSearchConfig(),
        kg_config=NetworkxKGConfig(),
        link_predictor_config=RandomLinkPredictorConfig(),
        score_aggregator_config=ProductScoreAggregatorConfig(),
        use_kg_ledger=False,
        use_filtering=False,
    )
    container.config.from_pydantic(config)

    container.wire(packages=["complex_query"])

    domains = ["domain_a", "domain_b"]

    kg = container.kg()
    for i in range(10):
        kg.add_concept(
            Concept(identifier=str(i), label=f"Node{i}", domain_identifier=domains[i % 2])
        )
    for i in range(10):
        for j in range(i + 1, 10):
            kg.add_triple(
                Triple(
                    subject_identifier=str(i),
                    relation_identifier="R" + str(10000 + i % 2),
                    object_identifier=str(j),
                )
            )

    request = QueryRequest(
        formula=(
            "(P(x, R10000, '8') AND P(x, R10001, '9')) OR (P(x, R10000, '6') AND P(x, R10001, '7'))"
        ),
        variables=[RequestVariable(label="x", domain=domains[0])],
        n_candidates=5,
    )
    answer_query(request)


def test_answer_query_different_scopes():
    container = QueryingContainer()
    config = QueryingSettings(
        parser_type=ParserType.LARK,
        search_config=BeamSearchConfig(),
        kg_config=NetworkxKGConfig(),
        link_predictor_config=RandomLinkPredictorConfig(),
        score_aggregator_config=ProductScoreAggregatorConfig(),
        use_kg_ledger=False,
        use_filtering=False,
    )
    container.config.from_pydantic(config)

    container.wire(packages=["complex_query"])

    domains = ["domain_a", "domain_b"]

    kg = container.kg()
    for i in range(10):
        kg.add_concept(
            Concept(identifier=str(i), label=f"Node{i}", domain_identifier=domains[i % 2])
        )
    for i in range(10):
        for j in range(i + 1, 10):
            kg.add_triple(
                Triple(
                    subject_identifier=str(i),
                    relation_identifier="R" + str(10000 + i % 2),
                    object_identifier=str(j),
                )
            )

    request = QueryRequest(
        formula=(
            "(P(x, 'R10000', \"8\") AND P(x, 'R10001', '9'))"
            " OR (P(x, 'R10000', '6') AND P(x, 'R10001', y))"
        ),
        variables=[
            RequestVariable(label="x", domain=domains[0]),
            RequestVariable(label="y", domain=domains[1]),
        ],
        n_candidates=5,
        search_algorithm="beam",
        search_parameters={"beam_size": 5},
    )
    answer_query(request)


def test_answer_query_with_more_candidates_than_beam_size():
    container = QueryingContainer()
    config = QueryingSettings(
        parser_type=ParserType.LARK,
        search_config=BeamSearchConfig(),
        kg_config=NetworkxKGConfig(),
        link_predictor_config=RandomLinkPredictorConfig(),
        score_aggregator_config=ProductScoreAggregatorConfig(),
        use_kg_ledger=False,
        use_filtering=False,
    )
    container.config.from_pydantic(config)

    container.wire(packages=["complex_query"])

    domains = ["domain_a", "domain_b"]

    kg = container.kg()
    for i in range(10):
        kg.add_concept(
            Concept(identifier=str(i), label=f"Node{i}", domain_identifier=domains[i % 2])
        )
    for i in range(10):
        for j in range(i + 1, 10):
            kg.add_triple(
                Triple(
                    subject_identifier=str(i),
                    relation_identifier="R" + str(10000 + i % 2),
                    object_identifier=str(j),
                )
            )

    request = QueryRequest(
        formula=(
            "(P(x, R10000, '8') AND P(x, R10001, '9')) OR (P(x, R10000, '6') AND P(x, R10001, '7'))"
        ),
        variables=[RequestVariable(label="x", domain=domains[0])],
        n_candidates=10,
        search_algorithm="beam",
        search_parameters={"beam_size": 5},
    )
    answer_query(request)
