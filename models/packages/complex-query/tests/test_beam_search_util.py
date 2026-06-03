from pathlib import Path
from unittest.mock import Mock

import pytest
from dotenv import load_dotenv

from complex_query.container import QueryingContainer, QueryingSettings
from complex_query.core.contracts.score_aggregator import ScoreAggregator
from complex_query.impl.search.beam_search.generic import (
    BeamSearch,
    PartialSolution,
    ProblemSimulator,
    Step,
)


@pytest.fixture
def container():
    load_dotenv(Path(__file__).parent / "test_container.env")

    config = QueryingSettings()
    container = QueryingContainer()
    container.config.from_pydantic(config)
    container.wire(packages=["complex_query"])
    yield container
    container.unwire()


class IncrementPartialSolution(PartialSolution):
    __test__ = False

    def __init__(self, value: int):
        self.value = value

    def __str__(self):
        return str(self.value)


class IncrementStep(Step):
    __test__ = False

    def __init__(self, increment: int):
        self.increment = increment

    def __str__(self):
        return f"Increment by {self.increment}"


class IncrementProblemSimulator(ProblemSimulator[IncrementPartialSolution, IncrementStep]):
    # The goal is to increment a state by values of 1, 2, 3 as quickly as possible to reach 10,
    # but without going over 10.
    __test__ = False

    def __init__(self, goal: int = 10):
        self.goal = goal

    def expand_solution(
        self,
        partial_solution: IncrementPartialSolution,
        step: IncrementStep,
        step_score: float,  # noqa: ARG002
    ) -> IncrementPartialSolution:
        new_value = partial_solution.value + step.increment
        return IncrementPartialSolution(new_value)

    def evaluate_step(
        self, partial_solution: IncrementPartialSolution, step: IncrementStep
    ) -> float:
        # Simple evaluation function: negative score for going over 10
        if partial_solution.value + step.increment > self.goal:
            return -10
        return step.increment

    def possible_steps_from_solution(
        self,
        partial_solution: IncrementPartialSolution,  # noqa: ARG002
    ) -> list[IncrementStep]:
        # Define steps as increments by 1, 2, and 3
        return [IncrementStep(1), IncrementStep(2), IncrementStep(3)]

    def is_solution_complete(self, partial_solution: IncrementPartialSolution) -> bool:
        # Solution is complete if the value is 10 or more
        return partial_solution.value >= self.goal


class SumScoreAggregator(ScoreAggregator):
    def binary_t_norm(self, a: float, b: float) -> float:
        return a + b


def test_beam_search(container):
    with container.logger.override(Mock()):
        initial_solution = IncrementPartialSolution(0)
        problem_simulator = IncrementProblemSimulator(goal=10)
        beam_size = 3
        top_step_paths = [
            [IncrementStep(3), IncrementStep(3), IncrementStep(3), IncrementStep(1)],
            [IncrementStep(3), IncrementStep(3), IncrementStep(2), IncrementStep(2)],
            [IncrementStep(3), IncrementStep(2), IncrementStep(3), IncrementStep(2)],
        ]
        top_step_paths_as_str = [",".join([str(step) for step in p]) for p in top_step_paths]
        stop_at_first_final_solution = True
        beam_search = BeamSearch[
            IncrementProblemSimulator, IncrementPartialSolution, IncrementStep
        ](
            problem_simulator,
            beam_size,
            stop_at_first_final_solution,
            score_aggregator=SumScoreAggregator(None),
        )
        final_nodes = beam_search.search(initial_solution)
        assert len(final_nodes) == 3
        assert final_nodes[0].cumulative_score == 10
        assert beam_search.score_aggregator.t_norm(final_nodes[0].get_score_path()) == 10
        assert beam_search.score_aggregator.t_conorm(final_nodes[0].get_score_path()) == 7
        assert (
            ",".join([str(b_step.step) for b_step in final_nodes[0].get_step_path()])
            in top_step_paths_as_str
        )

        stop_at_first_final_solution = False
        beam_search = BeamSearch(
            problem_simulator,
            beam_size,
            stop_at_first_final_solution,
            score_aggregator=SumScoreAggregator(None),
        )
        final_nodes = beam_search.search(initial_solution)
        assert len(final_nodes) == 3
        for node in final_nodes:
            assert node.cumulative_score == 10
            assert beam_search.score_aggregator.t_norm(node.get_score_path()) == 10
            assert beam_search.score_aggregator.t_conorm(final_nodes[0].get_score_path()) == 7
            assert (
                ",".join([str(b_step.step) for b_step in node.get_step_path()])
                in top_step_paths_as_str
            )
