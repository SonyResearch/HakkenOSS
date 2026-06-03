from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

PartialSolutionT = TypeVar("PartialSolutionT", bound="PartialSolution")
StepT = TypeVar("StepT", bound="Step")
ProblemSimulatorT = TypeVar("ProblemSimulatorT", bound="ProblemSimulator[Any, Any]")


class PartialSolution(ABC):
    """Partial solution to a problem"""

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    def short_repr(self) -> str:
        """Optional shorter string representation (for logging, visualization, etc., purposes)"""
        return str(self)


class Step(ABC):
    """Step to get from a partial solution to another"""

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    def short_repr(self) -> str:
        """Optional shorter string representation (for logging, visualization, etc., purposes)"""
        return str(self)


class ProblemSimulator(ABC, Generic[PartialSolutionT, StepT]):
    """Simulator of a problem, to move between and evaluate steps and solutions."""

    @abstractmethod
    def expand_solution(
        self, partial_solution: PartialSolutionT, step: StepT, step_score: float
    ) -> PartialSolutionT:
        raise NotImplementedError

    @abstractmethod
    def evaluate_step(self, partial_solution: PartialSolutionT, step: StepT) -> float:
        raise NotImplementedError

    def evaluate_batch_of_steps(
        self, partial_solution: PartialSolutionT, steps: list[StepT]
    ) -> list[float]:
        return [self.evaluate_step(partial_solution, step) for step in steps]

    @abstractmethod
    def possible_steps_from_solution(self, partial_solution: PartialSolutionT) -> list[StepT]:
        raise NotImplementedError

    @abstractmethod
    def is_solution_complete(self, partial_solution: PartialSolutionT) -> bool:
        raise NotImplementedError
