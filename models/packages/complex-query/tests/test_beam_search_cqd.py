import itertools
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv
from filtering.core.entities.candidate import (
    OutputNodeCandidate as FilteringOutputNodeCandidate,
)
from query_common.entities.conditions.link import LinkCondition
from query_common.entities.conditions.logical import NegatedCondition
from query_common.entities.kg.concept import Concept
from query_common.entities.kg.relation import Relation
from query_common.entities.kg.triple import Triple
from query_common.entities.query import Candidate
from query_common.entities.variable import Variable

from complex_query.container import QueryingContainer, QueryingSettings
from complex_query.core.entities.config.score_ledger import InMemoryScoreLedgerConfig
from complex_query.core.values.errors import (
    SearchInputError,
    SearchLogicError,
)
from complex_query.impl.score_ledger.in_memory import InMemoryScoreLedger
from complex_query.impl.search.beam_search import (
    QueryConditionStep,
    QueryPartialSolution,
    QuerySimulator,
    get_execution_order,
)

if TYPE_CHECKING:
    from query_common.entities.kg.identifier import DomainIdentifier

# Concept, Domain, Relation IDS
CID0, CID1, CID2, CID3, CID4, CID5 = "0", "1", "2", "3", "4", "5"
RID0, RID1 = "REL_0", "REL_1"

X = "domain1"
Y = "domain2"
C = Concept(identifier=CID0)
R = Relation(identifier=RID0)
R2 = Relation(identifier=RID1)
var_x = Variable(label="x", domain_identifier=X)
var_y = Variable(label="y", domain_identifier=Y)
r_x_y = LinkCondition(id_=0, subject=var_x, relation=R, object_=var_y)
r_x_c = LinkCondition(id_=1, subject=var_x, relation=R, object_=C)
r_y_c = LinkCondition(id_=2, subject=var_y, relation=R, object_=C)
r2_x_y = LinkCondition(id_=3, subject=var_x, relation=R2, object_=var_y)
not_r_x_c = NegatedCondition(id_=4, condition=r_x_c)
conjunctive_conditions_3 = [r_x_y, not_r_x_c, r_y_c]
non_anchored_conditions = [r_x_y]


def mock_kg_concept_retrieval(domain: "DomainIdentifier") -> list[Concept]:
    if domain == X:
        return [Concept(identifier=CID1), Concept(identifier=CID2), Concept(identifier=CID3)]
    if domain == Y:
        return [Concept(identifier=CID4), Concept(identifier=CID5)]
    raise ValueError(f"Domain {domain} not in the KG")


@pytest.fixture
def container():
    load_dotenv(Path(__file__).parent / "test_container.env")

    config = QueryingSettings()
    container = QueryingContainer()
    container.config.from_pydantic(config)
    container.wire(packages=["complex_query"])
    yield container
    container.unwire()


@pytest.mark.parametrize(
    "conditions, correct_order",
    [
        ([r_x_y, r_x_c, r_y_c], [1, 0, 2]),
        ([r_x_y, r2_x_y, r_y_c], [2, 0, 1]),
        ([r_x_y, not_r_x_c, r_y_c], [2, 0, 1]),
    ],
)
def test_get_execution_order(conditions: list[LinkCondition], correct_order: list[int]):
    """Correct order is a list such that correct_order[i] gives the condition to be executed."""
    execution_order = get_execution_order(conditions)
    for i in range(len(execution_order)):
        assert execution_order[i] == correct_order[i]


class TestQuerySimulator:
    @pytest.fixture
    def mock_kg(self):
        mock = MagicMock()
        mock.get_concepts_from_domain = MagicMock(side_effect=mock_kg_concept_retrieval)
        return mock

    @pytest.fixture
    def mock_link_predictor(self):
        return MagicMock()

    @pytest.fixture
    def mock_node_filtering(self):
        mock = MagicMock()
        mock.filter = lambda candidates: [
            FilteringOutputNodeCandidate(node_id=input_candidate.node_id, filter_score=0.5)
            for input_candidate in candidates
        ]
        return mock

    @pytest.fixture
    def query_simulator(self, mock_kg, mock_link_predictor, mock_node_filtering):
        return QuerySimulator(
            kg=mock_kg,
            ledger=InMemoryScoreLedger(InMemoryScoreLedgerConfig()),
            link_predictor=mock_link_predictor,
            conjunctive_conditions=conjunctive_conditions_3,
            node_filtering=mock_node_filtering,
        )

    def test_evaluate_step(self, query_simulator, mock_link_predictor):
        partial_solution = QueryPartialSolution.from_empty()
        step = QueryConditionStep(condition=r_x_y, assignment={"x": CID2, "y": CID3})

        # predict link score success
        mock_link_predictor.predict.return_value = 0.7
        score = query_simulator.evaluate_step(partial_solution, step)
        assert score == 0.7

        # retrieve link score success
        mock_link_predictor.reset_mock()
        score = query_simulator.evaluate_step(partial_solution, step)
        assert score == 0.7
        mock_link_predictor.predict.assert_not_called()

    @patch("complex_query.impl.search.beam_search.QuerySimulator.get_next_condition_to_run")
    def test_possible_steps_from_empty_solution(
        self, mock_get_next_condition, query_simulator, container
    ):
        with container.logger.override(MagicMock()):
            partial_solution = QueryPartialSolution.from_empty()

            # R(x,c)  # noqa: ERA001
            mock_get_next_condition.return_value = r_x_c
            steps = query_simulator.possible_steps_from_solution(partial_solution)
            assert len(steps) == 3
            assert all(s.condition == r_x_c for s in steps)
            assert {s.assignment[var_x.label] for s in steps} == {CID1, CID2, CID3}

            # R(x,y)  # noqa: ERA001
            mock_get_next_condition.return_value = r_x_y
            steps = query_simulator.possible_steps_from_solution(
                partial_solution, ignore_search_space_exception=True
            )
            assert len(steps) == 3 * 2
            assert all(s.condition == r_x_y for s in steps)
            assert {(s.assignment[var_x.label], s.assignment[var_y.label]) for s in steps} == set(
                itertools.product([CID1, CID2, CID3], [CID4, CID5])
            )

    @patch("complex_query.impl.search.beam_search.QuerySimulator.get_next_condition_to_run")
    def test_possible_steps_from_solution_for_non_anchored_query(
        self,
        mock_get_next_condition,
        mock_kg,
        mock_link_predictor,
        mock_node_filtering,
        container,
    ):
        with container.logger.override(MagicMock()):
            query_simulator = QuerySimulator(
                conjunctive_conditions=non_anchored_conditions,
                kg=mock_kg,
                link_predictor=mock_link_predictor,
                ledger=InMemoryScoreLedger(InMemoryScoreLedgerConfig()),
                node_filtering=mock_node_filtering,
            )
            partial_solution = QueryPartialSolution.from_empty()
            mock_get_next_condition.return_value = r_x_y
            with pytest.raises(SearchInputError):
                query_simulator.possible_steps_from_solution(partial_solution)
            partial_solution = QueryPartialSolution(
                Candidate.model_construct(
                    var_assignments={var_x.label: CID1}, condition_scores=MagicMock()
                )
            )
            steps = query_simulator.possible_steps_from_solution(partial_solution)
            assert len(steps) == len(mock_kg_concept_retrieval(var_y.domain_identifier))

    @patch("complex_query.impl.search.beam_search.QuerySimulator.get_next_condition_to_run")
    def test_possible_steps_from_non_empty_solution(
        self, mock_get_next_condition, query_simulator, container
    ):
        with container.logger.override(MagicMock()):
            partial_solution = QueryPartialSolution(
                Candidate.model_construct(
                    var_assignments={var_x.label: CID1}, condition_scores=MagicMock()
                )
            )

            # R(x,c)  # noqa: ERA001
            mock_get_next_condition.return_value = r_x_c
            steps = query_simulator.possible_steps_from_solution(partial_solution)
            assert len(steps) == 1
            assert steps[0].assignment[var_x.label] == CID1

            # R(x,y)  # noqa: ERA001
            mock_get_next_condition.return_value = r_x_y
            steps = query_simulator.possible_steps_from_solution(partial_solution)
            assert len(steps) == 2
            assert {(s.assignment[var_x.label], s.assignment[var_y.label]) for s in steps} == set(
                itertools.product([CID1], [CID4, CID5])
            )

    def test_batch_prediction(self, query_simulator, container, mock_link_predictor):
        with container.logger.override(MagicMock()):
            partial_solution = QueryPartialSolution.from_empty()
            # batch prediction
            steps = [
                QueryConditionStep(condition=r_x_y, assignment={"x": CID2, "y": CID0}),
                QueryConditionStep(condition=r_x_y, assignment={"x": CID2, "y": CID1}),
                QueryConditionStep(condition=r_x_y, assignment={"x": CID2, "y": CID2}),
            ]
            mock_link_predictor.predict_batch.return_value = [0.6, 0.9, 0.0]
            scores = query_simulator.evaluate_batch_of_steps(partial_solution, steps)
            assert scores == [0.6, 0.9, 0.0]
            # batch with retrieval
            steps = [
                QueryConditionStep(condition=r_x_y, assignment={"x": CID0, "y": CID1}),
                QueryConditionStep(
                    condition=r_x_y, assignment={"x": CID2, "y": CID1}
                ),  # same as 2nd above
                QueryConditionStep(condition=r_x_y, assignment={"x": CID4, "y": CID1}),
            ]
            mock_link_predictor.predict_batch.return_value = [
                0.2,
                1.0,
            ]  # result for 1st and 3rd
            scores = query_simulator.evaluate_batch_of_steps(partial_solution, steps)
            assert scores == [0.2, 0.9, 1.0]

    def test_evaluate_and_run_steps_until_solution_is_complete(
        self, query_simulator, mock_link_predictor, container
    ):
        with container.logger.override(MagicMock()):
            mock_link_predictor.predict.return_value = 1.0
            solution = QueryPartialSolution.from_empty()
            for _ in range(len(query_simulator.conjunctive_conditions)):
                assert not query_simulator.is_solution_complete(solution)
                # evaluate all steps from current condition in order
                steps = query_simulator.possible_steps_from_solution(solution)
                for step in steps:
                    score = query_simulator.evaluate_step(solution, step)
                # run one step
                solution = query_simulator.expand_solution(solution, step, score)
            assert query_simulator.is_solution_complete(solution)
            with pytest.raises(SearchLogicError):
                steps = query_simulator.possible_steps_from_solution(solution)

    def test_expand_solution(self, query_simulator):
        # Expanding an empty solution
        solution = QueryPartialSolution.from_empty()
        step1 = QueryConditionStep(
            condition=r_x_c, assignment={var_x.label: CID1, C.label: C.identifier}
        )
        solution = query_simulator.expand_solution(solution, step1, 0.8)
        assert all(
            solution.candidate.var_assignments[v_label] == value
            for v_label, value in step1.assignment.items()
        )
        assert solution.candidate.condition_scores[r_x_c.id] == 0.8

        step2 = QueryConditionStep(
            condition=r_x_y, assignment={var_x.label: CID1, var_y.label: CID2}
        )
        # Expanding a non-empty solution, with an assignment that fits
        solution = query_simulator.expand_solution(solution, step2, 0.6)
        assert all(
            solution.candidate.var_assignments[v_label] == value
            for v_label, value in step2.assignment.items()
        )
        assert all(
            solution.candidate.var_assignments[v_label] == value
            for v_label, value in step1.assignment.items()
        )
        assert solution.candidate.condition_scores[r_x_c.id] == 0.8
        assert solution.candidate.condition_scores[r_x_y.id] == 0.6

        # Expanding a non-empty solution, with an assignment that doesn't fit
        with pytest.raises(SearchLogicError):
            step3 = QueryConditionStep(
                condition=r2_x_y, assignment={var_x.label: CID1, var_y.label: CID3}
            )
            solution = query_simulator.expand_solution(solution, step3, 0.6)

        # Expanding a non-empty solution, with a condition already in it
        with pytest.raises(SearchLogicError):
            step4 = QueryConditionStep(
                condition=r_x_y, assignment={var_x.label: CID1, var_y.label: CID2}
            )
            solution = query_simulator.expand_solution(solution, step4, 0.6)

    def test_ground_triple(self):
        triple = QuerySimulator.ground_triple_using_assignment(
            r_x_y, assignment={var_x.label: CID0, var_y.label: CID2}
        )
        assert triple == Triple(
            subject_identifier=CID0, relation_identifier=R.identifier, object_identifier=CID2
        )
        with pytest.raises(KeyError):  # Missing assignment for y
            QuerySimulator.ground_triple_using_assignment(r_x_y, assignment={var_x.label: CID0})
        triple = QuerySimulator.ground_triple_using_assignment(
            r_x_c, assignment={var_x.label: CID3}
        )
        assert triple == Triple(
            subject_identifier=CID3, relation_identifier=R.identifier, object_identifier=CID0
        )
