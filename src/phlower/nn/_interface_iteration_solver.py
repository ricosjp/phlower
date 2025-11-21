from __future__ import annotations

import abc

from phlower.collections import IPhlowerTensorCollections
from phlower.settings._iteration_solver_setting import (
    IPhlowerIterationSolverSetting,
)


class IOptimizeProblem(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step_forward(
        self, h: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        """Objective function to minimize

        min f(x)
         x

        Args:
            h (IPhlowerTensorCollections): collection of inputs

        Returns:
            IPhlowerTensorCollections: collection of outputs
        """
        ...

    @abc.abstractmethod
    def gradient(
        self,
        h: IPhlowerTensorCollections,
        update_keys: list[str],
        operator_keys: list[str] | None = None,
    ) -> IPhlowerTensorCollections:
        """gradient value of optimize problem

        Return \nabla f

        Args:
            h (IPhlowerTensorCollections): collection of inputs
            update_keys (list[str]): keys of update variables
            operator_keys (list[str]): keys of operator variables.
                If fed, the values of the keys are directly used as the
                output values of the operator, i.e., D[u] = NN[u].
                If not, compute values of update keys subtracted from the
                initial values to obtain the output values of the
                operator, i.e., D[u] = NN[u] - u.

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
