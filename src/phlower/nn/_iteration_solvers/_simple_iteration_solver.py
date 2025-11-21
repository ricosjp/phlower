from __future__ import annotations

import torch

from phlower.collections import IPhlowerTensorCollections
from phlower.nn._interface_iteration_solver import (
    IFIterationSolver,
    IOptimizeProblem,
)
from phlower.settings._iteration_solver_setting import (
    IPhlowerIterationSolverSetting,
    SimpleSolverSetting,
)
from phlower.utils import get_logger

_logger = get_logger(__name__)


class SimpleIterationSolver(IFIterationSolver):
    _EPS = 1e-8

    @classmethod
    def from_setting(
        cls, setting: IPhlowerIterationSolverSetting
    ) -> IFIterationSolver:
        assert isinstance(setting, SimpleSolverSetting)
        return SimpleIterationSolver(**setting.model_dump())

    def __init__(
        self,
        max_iterations: int,
        convergence_threshold: float,
        update_keys: list[str],
    ) -> None:
        self._max_iterations = max_iterations
        self._convergence_threshold = convergence_threshold
        self._targets = update_keys
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
        for i in range(self._max_iterations):
            h_next = problem.step_forward(h)

            diff_h = (h_next.mask(self._targets) - h.mask(self._targets)).apply(
                torch.linalg.norm
            )
            rel_diff_h = diff_h / (
                h.mask(self._targets).apply(torch.linalg.norm) + self._EPS
            )
            h.update(h_next.mask(self._targets), overwrite=True)

            if rel_diff_h < self._convergence_threshold:
                _logger.info(
                    f"Simple iteration solver is converged. ({i} iteration)"
                )
                self._is_converged = True
                break

        return h.mask(self._targets)
