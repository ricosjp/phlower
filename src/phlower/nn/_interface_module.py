from __future__ import annotations

import abc
import pathlib
from collections.abc import Callable
from typing import Generic, Self, TypeVar

from phlower_tensor import ISimulationField, PhlowerTensor
from phlower_tensor.collections import IPhlowerTensorCollections

from phlower.settings._module_settings import IPhlowerLayerParameters
from phlower.settings._preset_group_parameter_setting import (
    IPhlowerPresetGroupParameters,
)
from phlower.utils.calculation_state import CalculationState
from phlower.utils.typing import PhlowerForwardHook

T1 = TypeVar("T1", IPhlowerLayerParameters, IPhlowerPresetGroupParameters)
T2 = TypeVar("T2", PhlowerTensor, IPhlowerTensorCollections)


class IGenericPhlowerCoreModule(Generic[T1, T2], metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_setting(cls, setting: T1) -> Self: ...

    @classmethod
    @abc.abstractmethod
    def get_nn_name(cls) -> str: ...

    @classmethod
    @abc.abstractmethod
    def need_reference(cls) -> bool: ...

    @abc.abstractmethod
    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None: ...

    @abc.abstractmethod
    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None = None,
        state: CalculationState | None = None,
        **kwards,
    ) -> T2: ...

    @abc.abstractmethod
    def get_reference_name(self) -> str | None: ...


IPhlowerCoreModule = IGenericPhlowerCoreModule[
    IPhlowerLayerParameters, PhlowerTensor
]
IPhlowerCorePresetGroupModule = IGenericPhlowerCoreModule[
    IPhlowerPresetGroupParameters, IPhlowerTensorCollections
]


class IReadonlyReferenceGroup(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def search_module(self, name: str) -> IPhlowerCoreModule: ...

    @abc.abstractmethod
    def get_unique_name(self) -> str:
        """
        Get the name of the group.
        This name is unique across the entire module tree.
        e.g. "ModuleA.ModuleB" for a module C inside module B.
        """
        ...


class IPhlowerGroup(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step_forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField,
        **kwards,
    ) -> IPhlowerTensorCollections: ...


class IPhlowerModuleAdapter(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
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
    def get_n_nodes(self) -> list[int] | None: ...

    @abc.abstractmethod
    def get_display_info(self) -> str: ...

    @abc.abstractmethod
    def draw(self, output_directory: pathlib.Path, recursive: bool): ...

    @abc.abstractmethod
    def get_core_module(
        self,
    ) -> IGenericPhlowerCoreModule: ...

    @abc.abstractmethod
    def register_phlower_forward_hook(
        self, hook: PhlowerForwardHook
    ) -> None: ...

    @abc.abstractmethod
    def register_phlower_finalize_debug_hook(
        self, hook: Callable[[], None]
    ) -> None: ...

    @abc.abstractmethod
    def finalize_debug(self) -> None:
        """
        Run finalize debug hooks.
        If no finalize debug hook is registered, this method does nothing.
        """
        ...
