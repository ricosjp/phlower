from types import TracebackType

import torch
from typing_extensions import Self

from phlower._fields._simulation_field import ISimulationField
from phlower.collections import IPhlowerTensorCollections
from phlower.nn._interface_module import IPhlowerGroup
from phlower.nn._iteration_solver import IFIterationSolver
from phlower.utils.calculations import calculate_residual


class SimpleIterationSolver(IFIterationSolver):
    def __init__(self) -> None:
        self._max_iterations = 10
        self._criteria: float = 0.0
        self._keys = []
        self._residuals: torch.Tensor = torch.Tensor(0.0, dtype=torch.float32)

    def __enter__(self) -> Self:
        self._residuals *= 0.0
        return self

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ):
        return None

    def get_converged(self) -> bool: ...

    def run(
        self,
        group: IPhlowerGroup,
        *,
        inputs: IPhlowerTensorCollections,
        field_data: ISimulationField,
        **kwards,
    ) -> IPhlowerTensorCollections:
        h = inputs
        for _ in self._max_iterations:
            h_next = group.step_forward(h, field_data=field_data, **kwards)
            self._residuals = calculate_residual(h, h_next, keys=self._keys)
            h = h_next

            if self.get_converged():
                break

        return h
