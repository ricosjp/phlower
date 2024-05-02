from __future__ import annotations

import dagstream
import torch

from phlower import PhlowerTensor
from phlower.collections.tensors import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.nn._interface_layer import (
    IPhlowerCoreModule,
    IPhlowerModuleAdapter,
)


class PhlowerModuleAdapter(IPhlowerModuleAdapter, torch.nn.Module):
    """This layer handles common attributes for all PhlowerLayers.
    Example: no_grad
    """

    @classmethod
    def from_setting(
        cls, layer: IPhlowerCoreModule, setting
    ) -> PhlowerModuleAdapter:
        return cls(layer, **setting.__dict__)

    def __init__(
        self, layer: IPhlowerCoreModule, no_grad: bool = False
    ) -> None:
        super().__init__()
        self._layer = layer
        self._name: str = ""
        self._no_grad = no_grad
        self._input_keys: list[str] = []
        self._destinations = []
        self._output_key: str = []

    def get_destinations(self) -> list[str]:
        return self._destinations

    def construct(self) -> None: ...

    @property
    def name(self) -> str:
        return self._name

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor],
    ) -> IPhlowerTensorCollections:

        inputs = {key: data[key] for key in self._input_keys}
        if self._no_grad:
            with torch.no_grad():
                result = self._layer.forward(inputs, supports=supports)
        else:
            result = self._layer.forward(inputs, supports=supports)

        return phlower_tensor_collection({self._output_key: result})


class PhlowerGroupNetworks(IPhlowerModuleAdapter, torch.nn.Module):
    def __init__(self, modules: list[PhlowerModuleAdapter]) -> None:
        super().__init__()
        self._phlower_modules = modules
        self._no_grad = False
        self._input_keys: list[str] = []
        self._destinations = []
        self._output_keys: list[str] = []

        self._dag_modules = self.construct()

    def construct(self):
        for module in self._phlower_modules:
            module.construct()

        # topological-sort
        stream = dagstream.DagStream()
        _nodes = stream.emplace(*self._phlower_modules)
        for i, _emplace in enumerate(_nodes):
            _emplace.display_name = self._phlower_modules[i].name

        # ordering
        name2nodes = {node.display_name: node for node in _nodes}
        for module in self._phlower_modules:
            dest_names = module.get_destinations()
            name2nodes[module.name].precede(
                *[name2nodes[name] for name in dest_names], pipe=True
            )

        # construct (and check dags)
        dag_modules = stream.construct()

        return dag_modules

    def draw(self): ...

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor],
    ) -> IPhlowerTensorCollections:

        results = {}

        while self._dag_modules.is_active:
            nodes = self._dag_modules.get_ready()
            for node in nodes:
                if node.n_predecessors == 0:
                    inputs = phlower_tensor_collection(
                        {key: data[key] for key in self._input_keys}
                    )
                    node.receive_args(inputs)

                args = node.get_received_args()
                _module: PhlowerModuleAdapter = node.get_user_function()

                _result = _module.forward(args, supports=supports)

                self._dag_modules.send(node.mut_name, _result)
                self._dag_modules.done(node.mut_name)

                if self._dag_modules.check_last(node):
                    results.update(_result)

        return phlower_tensor_collection(results)
