from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from phlower_tensor import ISimulationField
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn._core_modules import get_module
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IPhlowerModuleAdapter,
    IReadonlyReferenceGroup,
)
from phlower.nn._utils import attach_location_to_error_message
from phlower.settings._group_setting import ModuleSetting
from phlower.utils import get_logger
from phlower.utils.calculation_state import CalculationState
from phlower.utils.typing import PhlowerForwardHook

from ._utils import attach_debug_helper

_logger = get_logger(__name__)


class PhlowerModuleAdapter(torch.nn.Module, IPhlowerModuleAdapter):
    """This layer handles common attributes for all PhlowerLayers.
    Example: no_grad
    """

    @classmethod
    def from_setting(cls, setting: ModuleSetting) -> PhlowerModuleAdapter:
        layer = get_module(setting.nn_type).from_setting(setting.nn_parameters)
        mod = PhlowerModuleAdapter(
            layer,
            name=setting.name,
            input_keys=setting.input_keys,
            output_key=setting.output_key,
            destinations=setting.destinations,
            no_grad=setting.no_grad,
            n_nodes=setting.nn_parameters.get_n_nodes(),
            coeff=setting.coeff,
        )
        attach_debug_helper(mod, setting.debug_parameters)
        return mod

    def __init__(
        self,
        layer: IPhlowerCoreModule,
        name: str,
        input_keys: list[str],
        output_key: str,
        destinations: list[str],
        no_grad: bool,
        n_nodes: list[int],
        coeff: float,
        **kwards,
    ) -> None:
        super().__init__(**kwards)
        self._layer = layer
        self._name = name
        self._no_grad = no_grad
        self._input_keys = input_keys
        self._destinations = destinations
        self._output_key = output_key
        self._n_nodes = n_nodes
        self._coeff = coeff

        self._forward_post_hooks: list[PhlowerForwardHook] = []
        self._finalize_hooks: list[Callable[[], None]] = []
        self._parent_prefix = ""

    def get_destinations(self) -> list[str]:
        return self._destinations

    def get_unique_name(self) -> str:
        return self._parent_prefix + self._name

    def register_phlower_forward_hook(self, hook: PhlowerForwardHook) -> None:
        self._forward_post_hooks.append(hook)

    def register_phlower_finalize_debug_hook(
        self, hook: Callable[[], None]
    ) -> None:
        self._finalize_hooks.append(hook)

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None, **kwards
    ) -> None:
        self._layer.resolve(parent=parent, **kwards)
        if parent is not None:
            self._parent_prefix = parent.get_unique_name() + "."

    def get_n_nodes(self) -> list[int]:
        return self._n_nodes

    def get_display_info(self) -> str:
        if not self._layer.need_reference():
            return (
                f"nn_type: {self._layer.get_nn_name()}\n"
                f"n_nodes: {self.get_n_nodes()}"
            )

        return (
            f"nn_type: {self._layer.get_nn_name()}\n"
            f"n_nodes: {self.get_n_nodes()} \n"
            f"reference: {self._layer.get_reference_name()}"
        )

    def draw(self, output_directory: Path, recursive: bool): ...

    @property
    def name(self) -> str:
        return self._name

    @attach_location_to_error_message
    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None = None,
        state: CalculationState | None = None,
        **kwards,
    ) -> IPhlowerTensorCollections:
        inputs = phlower_tensor_collection(
            {key: data[key] for key in self._input_keys}
        )
        if self._no_grad:
            with torch.no_grad():
                result = self._coeff * self._layer.forward(
                    inputs, field_data=field_data, state=state, **kwards
                )
        else:
            result = self._coeff * self._layer.forward(
                inputs, field_data=field_data, state=state, **kwards
            )

        ret = phlower_tensor_collection({self._output_key: result})

        for hook in self._forward_post_hooks:
            hook(
                name=self._name,
                unique_name=self.get_unique_name(),
                result=ret,
                state=state,
            )
        return ret

    def get_core_module(self) -> IPhlowerCoreModule:
        return self._layer

    def finalize_debug(self) -> None:
        for hook in self._finalize_hooks:
            hook()
