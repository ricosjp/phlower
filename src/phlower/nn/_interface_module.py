import abc
import pathlib

from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.settings._interface import IPhlowerLayerParameters


class IPhlowerCoreModule(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def from_setting(cls, setting: IPhlowerLayerParameters) -> Self: ...

    @abc.abstractclassmethod
    def get_parameter_setting_cls(cls) -> IPhlowerLayerParameters: ...

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
    def resolve(self) -> None: ...

    @abc.abstractmethod
    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor]
    ) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def get_destinations(self) -> list[str]: ...

    @abc.abstractmethod
    def get_n_nodes(self) -> list[int]: ...

    @abc.abstractmethod
    def get_display_info(self) -> str: ...

    @abc.abstractmethod
    def draw(self, output_directory: pathlib.Path, recursive: bool): ...
