from __future__ import annotations

from pathlib import Path

import torch

from phlower import PhlowerTensor
from phlower.collections.tensors import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.nn._core_modules import get_module
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IPhlowerModuleAdapter,
)
from phlower.settings._model_settings import ModuleSetting


class PhlowerModuleAdapter(IPhlowerModuleAdapter, torch.nn.Module):
    """This layer handles common attributes for all PhlowerLayers.
    Example: no_grad
    """

    @classmethod
    def from_setting(cls, setting: ModuleSetting) -> PhlowerModuleAdapter:
        layer = get_module(setting.nn_type).from_setting(setting.nn_parameters)
        return cls(
            layer,
            name=setting.name,
            input_keys=setting.input_keys,
            output_key=setting.output_key,
            destinations=setting.destinations,
            no_grad=setting.no_grad,
            n_nodes=setting.nn_parameters.get_n_nodes(),
        )

    def __init__(
        self,
        layer: IPhlowerCoreModule,
        name: str,
        input_keys: list[str],
        output_key: str,
        destinations: list[str],
        no_grad: bool,
        n_nodes: list[int],
    ) -> None:
        super().__init__()
        self._layer = layer
        self._name = name
        self._no_grad = no_grad
        self._input_keys = input_keys
        self._destinations = destinations
        self._output_key = output_key
        self._n_nodes = n_nodes

    def get_destinations(self) -> list[str]:
        return self._destinations

    def resolve(self) -> None: ...

    def get_n_nodes(self) -> list[int]:
        return self._n_nodes

    def get_display_info(self) -> str:
        return (
            f"nn_type: {self._layer.get_nn_name()}\n"
            f"n_nodes: {self.get_n_nodes()}"
        )

    def draw(self, output_directory: Path, recursive: bool): ...

    @property
    def name(self) -> str:
        return self._name

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor],
    ) -> IPhlowerTensorCollections:

        inputs = phlower_tensor_collection(
            {key: data[key] for key in self._input_keys}
        )
        if self._no_grad:
            with torch.no_grad():
                result = self._layer.forward(inputs, supports=supports)
        else:
            result = self._layer.forward(inputs, supports=supports)

        return phlower_tensor_collection({self._output_key: result})
