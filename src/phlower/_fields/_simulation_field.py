import abc
from collections.abc import Iterable

from phlower._base import GraphBatchInfo, PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections


class ISimulationField(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __getitem__(self, name: str) -> PhlowerTensor: ...

    @abc.abstractmethod
    def keys(self): ...

    @abc.abstractmethod
    def get_batch_info(self, name: str) -> GraphBatchInfo: ...

    @abc.abstractmethod
    def get_batched_n_nodes(self, name: str) -> list[int]: ...


class SimulationField(ISimulationField):
    def __init__(
        self,
        field_tensors: IPhlowerTensorCollections,
        batch_info: dict[str, GraphBatchInfo] | None = None,
    ) -> None:
        self._field_tensors = field_tensors

        if batch_info is None:
            batch_info = {}
        self._batch_info = batch_info

    def keys(self) -> Iterable[str]:
        return self._field_tensors.keys()

    def __getitem__(self, name: str) -> PhlowerTensor:
        if name not in self._field_tensors:
            raise KeyError(f"{name} is not found in simulation field.")
        return self._field_tensors[name]

    def get_batch_info(self, name: str) -> GraphBatchInfo:
        if name not in self._batch_info:
            raise KeyError(f"{name} is not found in simulation field.")
        return self._batch_info[name]

    def get_batched_n_nodes(self, name: str) -> list[int]:
        if not self._batch_info:
            raise ValueError("Information about batch is not found.")

        # NOTE: Assume that batch information is same among features.
        batch_info = self.get_batch_info(name)
        return batch_info.n_nodes

    # HACK: Under construction
    # def calculate_laplacians(self, target: PhlowerTensor):
    #     ...

    # def calculate_gradient(self, target: PhlowerTensor):
    #     ...
