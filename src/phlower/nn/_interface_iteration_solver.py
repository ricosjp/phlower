from __future__ import annotations

import abc

from phlower.collections import IPhlowerTensorCollections
from phlower.settings._nonlinear_solver_setting import (
    IPhlowerIterationSolverSetting,
)


class IOptimizeProblem:
    @abc.abstractmethod
    def objective(
        self, value: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        """Objective function to minimize

        min f(x)
         x

        Args:
            value (IPhlowerTensorCollections): collection of inputs

        Returns:
            IPhlowerTensorCollections: collection of outputs
        """
        ...

    @abc.abstractmethod
    def gradient(
        self, value: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        """gradient value of optimize problem

        Return \nabla f

        Args:
            value (IPhlowerTensorCollections): collection of inputs

        Returns:
            IPhlowerTensorCollections: collection of outputs
        """
        ...


class IFIterationSolver(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_setting(
        cls, setting: IPhlowerIterationSolverSetting
    ) -> IFIterationSolver: ...

    @abc.abstractmethod
    def get_converged(self) -> bool: ...

    @abc.abstractmethod
    def zero_residuals(self) -> None: ...

    @abc.abstractmethod
    def run(
        self,
        initial_values: IPhlowerTensorCollections,
        problem: IOptimizeProblem,
    ) -> IPhlowerTensorCollections: ...
