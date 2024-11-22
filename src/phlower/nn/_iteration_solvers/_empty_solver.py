from types import TracebackType

from typing_extensions import Self

from phlower._fields._simulation_field import ISimulationField
from phlower.collections import IPhlowerTensorCollections
from phlower.nn._interface_module import IPhlowerGroup
from phlower.nn._iteration_solver import IFIterationSolver


class EmptySolver(IFIterationSolver):
    def __init__(self) -> None: ...

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ):
        return None

    def run(
        self,
        group: IPhlowerGroup,
        *,
        inputs: IPhlowerTensorCollections,
        field_data: ISimulationField,
        **kwards,
    ) -> IPhlowerTensorCollections:
        return group.step_forward(inputs, field_data=field_data, **kwards)
