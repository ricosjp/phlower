from __future__ import annotations

import abc
from collections.abc import ItemsView, Iterable

import torch

from phlower._base import GraphBatchInfo, PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections


class ISimulationField(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __getitem__(self, name: str) -> PhlowerTensor: ...

    @abc.abstractmethod
    def __contains__(self, name: str) -> bool: ...

    @abc.abstractmethod
    def keys(self): ...

    @abc.abstractmethod
    def items(self): ...

    @abc.abstractmethod
    def get_batch_info(self, name: str) -> GraphBatchInfo: ...

    @abc.abstractmethod
    def get_batched_n_nodes(self, name: str) -> list[int]: ...

    @abc.abstractmethod
    def to(
        self, device: str | torch.device, non_blocking: bool = False
    ) -> ISimulationField: ...


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

    def items(self) -> ItemsView[str, PhlowerTensor]:
        return self._field_tensors.items()

    def __getitem__(self, name: str) -> PhlowerTensor:
        if name not in self._field_tensors:
            raise KeyError(f"{name} is not found in simulation field.")
        return self._field_tensors[name]

    def __contains__(self, name: str) -> bool:
        return name in self._field_tensors

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

    def to(
        self, device: str | torch.device, non_blocking: bool = False
    ) -> ISimulationField:
        return SimulationField(
            field_tensors=self._field_tensors.to(
                device, non_blocking=non_blocking
            ),
            batch_info=self._batch_info,
        )

    # HACK: Under construction
    # def calculate_laplacians(self, target: PhlowerTensor):
    #     ...

    # def calculate_gradient(self, target: PhlowerTensor):
    #     ...
