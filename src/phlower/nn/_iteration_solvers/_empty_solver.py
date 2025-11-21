from __future__ import annotations

from phlower.collections import IPhlowerTensorCollections
from phlower.nn._interface_iteration_solver import (
    IFIterationSolver,
    IOptimizeProblem,
)
from phlower.settings._iteration_solver_setting import (
    EmptySolverSetting,
    IPhlowerIterationSolverSetting,
)


class EmptySolver(IFIterationSolver):
    @classmethod
    def from_setting(
        cls, setting: IPhlowerIterationSolverSetting
    ) -> EmptySolver:
        assert isinstance(setting, EmptySolverSetting)
        return EmptySolver()

    def __init__(self) -> None: ...

    def zero_residuals(self) -> None:
        return

    def get_converged(self) -> bool:
        return True

    def run(
        self,
        initial_values: IPhlowerTensorCollections,
        problem: IOptimizeProblem,
    ) -> IPhlowerTensorCollections:
        return problem.step_forward(initial_values)
