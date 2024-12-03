from __future__ import annotations

import torch

from phlower.collections import IPhlowerTensorCollections
from phlower.nn._interface_iteration_solver import (
    IFIterationSolver,
    IOptimizeProblem,
)
from phlower.settings._nonlinear_solver_setting import (
    IPhlowerIterationSolverSetting,
    SimpleSolverSetting,
)


class SimpleIterationSolver(IFIterationSolver):
    @classmethod
    def from_setting(
        cls, setting: IPhlowerIterationSolverSetting
    ) -> IFIterationSolver:
        assert isinstance(setting, SimpleSolverSetting)
        return SimpleIterationSolver(**setting.__dict__)

    def __init__(
        self,
        max_iterations: int,
        divergence_threshold: float,
        target_keys: list[str],
    ) -> None:
        self._max_iterations = max_iterations
        self._divergence_threshold: float = divergence_threshold
        self._targets = target_keys
        self._is_converged = False

    def zero_residuals(self) -> None:
        self._is_converged = False
        return

    def get_converged(self) -> bool:
        return self._is_converged

    def run(
        self,
        initial_values: IPhlowerTensorCollections,
        problem: IOptimizeProblem,
    ) -> IPhlowerTensorCollections:
        h = initial_values
        for _ in range(self._max_iterations):
            h_next = problem.objective(h)

            diff_h = (h_next.mask(self._targets) - h.mask(self._targets)).apply(
                torch.linalg.norm
            )
            h.update(h_next.mask(self._targets), overwrite=True)

            if diff_h < self._divergence_threshold:
                self._is_converged = True
                break

        return h
