import abc
from types import TracebackType

from typing_extensions import Self

from phlower.collections import IPhlowerTensorCollections
from phlower.nn._interface_module import IPhlowerGroup, ISimulationField


class IFIterationSolver(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def get_converged(self) -> bool: ...

    @abc.abstractmethod
    def __enter__(self) -> Self: ...

    @abc.abstractmethod
    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ): ...

    @abc.abstractmethod
    def run(
        self,
        group: IPhlowerGroup,
        *,
        inputs: IPhlowerTensorCollections,
        field_data: ISimulationField,
        **kwards,
    ) -> IPhlowerTensorCollections: ...
