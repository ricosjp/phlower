from __future__ import annotations

import abc
import pathlib

from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.settings._module_settings import IPhlowerLayerParameters


class IPhlowerCoreModule(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_setting(cls, setting: IPhlowerLayerParameters) -> Self: ...

    @classmethod
    @abc.abstractmethod
    def get_nn_name(cls) -> str: ...

    @classmethod
    @abc.abstractmethod
    def need_resolve(cls) -> bool: ...

    @abc.abstractmethod
    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None: ...

    @abc.abstractmethod
    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor] = None,
    ) -> PhlowerTensor: ...


class IReadonlyReferenceGroup(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def search_module(self, name: str) -> IPhlowerModuleAdapter: ...


class IPhlowerModuleAdapter(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def resolve(
        cls, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None: ...

    @abc.abstractmethod
    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor] | None = None,
        **kwards,
    ) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def get_destinations(self) -> list[str]: ...

    @abc.abstractmethod
    def get_n_nodes(self) -> list[int]: ...

    @abc.abstractmethod
    def get_display_info(self) -> str: ...

    @abc.abstractmethod
    def draw(self, output_directory: pathlib.Path, recursive: bool): ...
