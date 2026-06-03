from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import tqdm
from dependency_injector.wiring import Provide, inject
from spaice_inference_api import ILogger, LoggerToken

from complex_query.impl.search.beam_search.generic.problem import (
    PartialSolutionT,
    ProblemSimulatorT,
    StepT,
)

if TYPE_CHECKING:
    from complex_query.core.contracts.score_aggregator import ScoreAggregator

BeamNodeT = TypeVar("BeamNodeT", bound="BeamNode[Any, Any]")
BeamStepT = TypeVar("BeamStepT", bound="BeamStep[Any, Any]")


#########################################
# Generic Beam Search for all problems. #
#########################################
@dataclass
class BeamStep(Generic[PartialSolutionT, StepT]):
    step: StepT
    score: float
    cumulative_score: float
    from_beam_node: BeamNode[PartialSolutionT, StepT]

    def __str__(self):
        return (
            f"Step: {self.step.short_repr()}, Score: {self.score},"
            f" Cumulative Score: {self.cumulative_score}"
        )


@dataclass
class BeamNode(Generic[PartialSolutionT, StepT]):
    partial_solution: PartialSolutionT
    cumulative_score: float
    is_final: bool
    from_beam_step: BeamStep[PartialSolutionT, StepT] | None = None
    """Beam step that originates a node"""
    beam_step_options: list[BeamStep[PartialSolutionT, StepT]] | None = None
    """Beam steps we could have taken (for logging purposes)"""

    def get_step_path(self) -> list[BeamStep[PartialSolutionT, StepT]]:
        """List of steps leading to the current beam node"""
        if self.from_beam_step is None:
            return []
        return [*self.from_beam_step.from_beam_node.get_step_path(), self.from_beam_step]

    def get_score_path(self) -> list[float]:
        """List of scores leading to the current beam node"""
        return [step.score for step in self.get_step_path()]

    def __str__(self):
        return (
            f"Partial solution: {self.partial_solution.short_repr()}, "
            f"Score: {self.cumulative_score}"
        )


class BeamSearch(Generic[ProblemSimulatorT, PartialSolutionT, StepT]):
    def __init__(  # noqa: PLR0913
        self,
        problem_simulator: ProblemSimulatorT,
        beam_size: int,
        stop_at_first_final_solution: bool,
        score_aggregator: ScoreAggregator,
        batch_steps: bool = True,
        batch_size: int = 32,
    ):
        self.problem_simulator = problem_simulator
        self.beam_size = beam_size
        self.stop_at_first_final_solution = stop_at_first_final_solution
        self.score_aggregator = score_aggregator
        self.batch_steps = batch_steps
        self.batch_size = batch_size

    @inject
    def search(
        self,
        initial_solution: PartialSolutionT,
        logger: ILogger = Provide[LoggerToken],
    ) -> list[BeamNode[PartialSolutionT, StepT]]:
        initial_node = BeamNode[PartialSolutionT, StepT](
            partial_solution=initial_solution,
            cumulative_score=0.0,
            is_final=self.problem_simulator.is_solution_complete(initial_solution),
        )
        logger.info(f"Starting from node {initial_node}.")
        current_beam: list[BeamNode[PartialSolutionT, StepT]] = [initial_node]
        search_step = 1
        while True:
            logger.info(f"Current beam size: {len(current_beam)}, Search step: {search_step}")
            search_step += 1
            next_beam = []
            for node in current_beam:
                logger.info(f"Next beam node -> Evaluating possible steps from {node}")
                if self.problem_simulator.is_solution_complete(node.partial_solution):
                    continue  # Stop exploring this node
                steps = self.problem_simulator.possible_steps_from_solution(node.partial_solution)
                if self.batch_steps:
                    batches_of_steps = [
                        steps[i : i + self.batch_size]
                        for i in range(0, len(steps), self.batch_size)
                    ]
                    scores_as_batches = [
                        self.problem_simulator.evaluate_batch_of_steps(node.partial_solution, batch)
                        for batch in tqdm.tqdm(
                            batches_of_steps,
                            desc=f"Evaluating batches of {self.batch_size} steps...",
                        )
                    ]
                    scores = list(chain(*scores_as_batches))
                else:
                    scores = [
                        self.problem_simulator.evaluate_step(node.partial_solution, step)
                        for step in tqdm.tqdm(steps, desc="Evaluating steps...")
                    ]
                # filter 1: beam_size steps from the current node
                steps_with_scores = list(zip(steps, scores, strict=False))
                steps_with_scores.sort(key=lambda pair: pair[1], reverse=True)
                top_steps_with_scores = steps_with_scores[: self.beam_size]
                logger.info(
                    f"Keeping top {self.beam_size} steps out of {len(scores)} options "
                    "and constructing the corresponding solutions."
                )
                beam_step_options = []
                for step, score in top_steps_with_scores:
                    cumulative_score = self.score_aggregator.t_norm([*node.get_score_path(), score])
                    new_partial_solution = self.problem_simulator.expand_solution(
                        node.partial_solution, step, score
                    )
                    beam_step = BeamStep[PartialSolutionT, StepT](
                        step=step,
                        score=score,
                        cumulative_score=cumulative_score,
                        from_beam_node=node,
                    )
                    beam_step_options.append(beam_step)
                    new_node = BeamNode[PartialSolutionT, StepT](
                        partial_solution=new_partial_solution,
                        cumulative_score=cumulative_score,
                        from_beam_step=beam_step,
                        is_final=self.problem_simulator.is_solution_complete(new_partial_solution),
                    )
                    next_beam.append(new_node)
                node.beam_step_options = beam_step_options
            # filter 2: beam_size steps from all nodes
            logger.info(
                f"Finished evaluating the best {self.beam_size} (max) steps from each of the "
                f"current {len(current_beam)} beam nodes."
                f" Now keeping the best {self.beam_size} candidates overall."
            )
            next_beam.sort(key=lambda x: x.cumulative_score, reverse=True)
            current_beam = next_beam[: self.beam_size]
            # check for ending conditions
            if self.stop_at_first_final_solution:
                for i, node in enumerate(current_beam):
                    if self.problem_simulator.is_solution_complete(node.partial_solution):
                        current_beam.insert(0, current_beam.pop(i))
                        logger.info(f"Early stopping at first final solution: {current_beam[0]}.")
                        break
            if all(
                self.problem_simulator.is_solution_complete(node.partial_solution)
                for node in current_beam
            ):
                beam_str_representation = "\n".join([str(beam_node) for beam_node in current_beam])
                logger.info(f"All beam nodes are complete solutions: {beam_str_representation}")
                logger.info("Finishing search.")
                break

        return sorted(current_beam, key=lambda x: x.cumulative_score, reverse=True)
