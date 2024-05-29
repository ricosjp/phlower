from __future__ import annotations

from typing_extensions import Self

import dagstream
from dagstream.viewers import MermaidDrawer
import torch

from phlower import PhlowerTensor
from phlower.settings._model_settings import GroupModuleSetting, ModuleSetting
from phlower.collections.tensors import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
    reduce_collections
)
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IPhlowerModuleAdapter,
)
from phlower.nn._core_modules import get_module


class PhlowerModuleAdapter(IPhlowerModuleAdapter, torch.nn.Module):
    """This layer handles common attributes for all PhlowerLayers.
    Example: no_grad
    """

    @classmethod
    def from_setting(
        cls, setting: ModuleSetting
    ) -> PhlowerModuleAdapter:
        layer = get_module(setting.nn_type).from_setting(setting.nn_parameters)
        return cls(layer, **setting.__dict__)

    def __init__(
        self, 
        layer: IPhlowerCoreModule,
        name: str,
        input_keys: list[str],
        output_key: str,
        destinations: list[str],
        no_grad: bool,
        **kwards
    ) -> None:
        super().__init__()
        self._layer = layer
        self._name: str = name
        self._no_grad = no_grad
        self._input_keys: list[str] = input_keys
        self._destinations = destinations
        self._output_key: str = output_key

    def get_destinations(self) -> list[str]:
        return self._destinations

    def resolve(self) -> None: ...

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


