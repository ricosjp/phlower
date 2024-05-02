import abc

from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections


class IPhlowerLayerParameters(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def gather_input_dims(cls, *input_dims: int) -> int: ...

    @abc.abstractmethod
    def get_nodes(self) -> list[int] | None: ...

    @abc.abstractmethod
    def overwrite_nodes(self, nodes: list[int]) -> None: ...


class IPhlowerCoreModule(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def from_setting(cls, setting: IPhlowerLayerParameters) -> Self: ...

    @abc.abstractclassmethod
    def get_nn_name(cls) -> str: ...

    @abc.abstractmethod
    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor] = None
    ) -> PhlowerTensor: ...


class IPhlowerModuleAdapter(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def construct(self) -> None: ...

    @abc.abstractmethod
    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor]
    ) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def get_destinations(self) -> list[str]: ...
