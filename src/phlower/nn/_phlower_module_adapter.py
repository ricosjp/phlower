from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from phlower._base import IPhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.nn._core_modules import get_module
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IPhlowerModuleAdapter,
    IReadonlyReferenceGroup,
)
from phlower.settings._debug_parameter_setting import (
    PhlowerModuleDebugParameters,
)
from phlower.settings._group_setting import ModuleSetting


class PhlowerModuleAdapter(torch.nn.Module, IPhlowerModuleAdapter):
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
            coeff=setting.coeff,
            debug_parameters=setting.debug_parameters,
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
        coeff: float,
        debug_parameters: PhlowerModuleDebugParameters,
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

        self._debug_helper = _DebugHelper(debug_parameters)

    def get_destinations(self) -> list[str]:
        return self._destinations

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None, **kwards
    ) -> None:
        if not self._layer.need_reference():
            return

        self._layer.resolve(parent=parent, **kwards)

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

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None = None,
    ) -> IPhlowerTensorCollections:
        inputs = phlower_tensor_collection(
            {key: data[key] for key in self._input_keys}
        )
        if self._no_grad:
            with torch.no_grad():
                result = self._coeff * self._layer.forward(
                    inputs, field_data=field_data
                )
        else:
            result = self._coeff * self._layer.forward(
                inputs, field_data=field_data
            )

        if self._debug_helper.need_debug():
            self._debug_helper.debug(self._name, result)

        return phlower_tensor_collection({self._output_key: result})

    def get_core_module(self) -> IPhlowerCoreModule:
        return self._layer


class _DebugHelper:
    def __init__(
        self,
        debug_parameters: PhlowerModuleDebugParameters,
    ):
        self._debug_parameters = debug_parameters

    def need_debug(self) -> bool:
        return self._debug_parameters.output_tensor_shape is not None

    def debug(self, name: str, result: IPhlowerTensor) -> None:
        if self._debug_parameters.output_tensor_shape:
            self._check_output_tensor_shape(name, result)

        return

    def _check_output_tensor_shape(
        self, name: str, result: IPhlowerTensor
    ) -> None:
        if len(self._debug_parameters.output_tensor_shape) != len(result.shape):
            raise ValueError(
                f"In {name}, result tensor shape "
                f"{result.shape} is different from desired shape "
                f"{self._debug_parameters.output_tensor_shape} "
                "in yaml file"
            )

        output_tensor_shape = np.array(
            self._debug_parameters.output_tensor_shape
        )
        positive_flag = output_tensor_shape > 0
        output_tensor_shape = np.where(
            positive_flag, output_tensor_shape, result.shape
        )
        if not all(output_tensor_shape == result.shape):
            ind = np.where(output_tensor_shape != result.shape)[0][0]

            raise ValueError(
                f"In {name}, {ind}-th result tensor shape "
                f"{result.shape} is different from desired shape "
                f"{output_tensor_shape} in yaml file"
            )
