from __future__ import annotations

import pathlib

import torch

from phlower._fields import ISimulationField
from phlower.collections.tensors import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.nn._core_preset_group_modules import (
    get_preset_group_module,
)
from phlower.nn._interface_module import (
    IPhlowerCorePresetGroupModule,
    IPhlowerModuleAdapter,
    IReadonlyReferenceGroup,
)
from phlower.settings._preset_group_setting import PresetGroupModuleSetting


class PhlowerPresetGroupModuleAdapter(
    torch.nn.Module,
    IPhlowerModuleAdapter,
):
    """This layer handles common attributes for all PhlowerLayers.
    Example: no_grad
    """

    @classmethod
    def from_setting(
        cls, setting: PresetGroupModuleSetting
    ) -> PhlowerPresetGroupModuleAdapter:
        layer = get_preset_group_module(setting.preset_type).from_setting(
            setting.nn_parameters
        )

        return PhlowerPresetGroupModuleAdapter(
            layer,
            name=setting.name,
            no_grad=setting.no_grad,
            input_keys=setting.get_input_keys(),
            output_keys=setting.get_output_keys(),
            destinations=setting.destinations,
            input_nodes=setting.inputs,
            output_nodes=setting.outputs,
        )

    def __init__(
        self,
        layer: IPhlowerCorePresetGroupModule,
        name: str,
        no_grad: bool,
        input_keys: list[str],
        destinations: list[str],
        output_keys: list[str],
        input_nodes: dict[str, int],
        output_nodes: dict[str, int],
    ) -> None:
        super().__init__()
        self._layer = layer
        self._name = name
        self._no_grad = no_grad
        self._input_keys = input_keys
        self._destinations = destinations
        self._output_keys = output_keys
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes

    def get_destinations(self) -> list[str]:
        return self._destinations

    @property
    def name(self) -> str:
        return self._name

    def get_display_info(self) -> str:
        return (
            f"input_keys: {self._input_keys}\noutput_keys: {self._output_keys}"
        )

    def get_n_nodes(self) -> list[int] | None:
        return None

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None:
        if not self._layer.need_reference():
            return

        self._layer.resolve(parent=parent, **kwards)

    def draw(
        self, output_directory: pathlib.Path | str, recursive: bool = True
    ): ...

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField,
        **kwards,
    ) -> IPhlowerTensorCollections:
        inputs = phlower_tensor_collection(
            {key: data[key] for key in self._input_keys}
        )
        if self._no_grad:
            with torch.no_grad():
                result = self._layer.forward(inputs, field_data=field_data)
        else:
            result = self._layer.forward(inputs, field_data=field_data)

        return result

    def get_core_module(self) -> IPhlowerCorePresetGroupModule:
        return self._layer
