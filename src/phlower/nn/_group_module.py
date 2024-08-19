from __future__ import annotations

import pathlib

import dagstream
import torch
from typing_extensions import Self

from phlower import PhlowerTensor
from phlower.collections.tensors import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
    reduce_collections,
)
from phlower.io._files import IPhlowerCheckpointFile
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IPhlowerModuleAdapter,
    IReadonlyReferenceGroup,
)
from phlower.nn._phlower_module_adpter import PhlowerModuleAdapter
from phlower.services.drawers import MermaidDrawer
from phlower.settings._group_settings import GroupModuleSetting, ModuleSetting
from phlower.utils.enums import TrainerSavedKeyType


class PhlowerGroupModule(
    IPhlowerModuleAdapter, IReadonlyReferenceGroup, torch.nn.Module
):
    @classmethod
    def from_setting(cls, setting: GroupModuleSetting) -> Self:
        _modules: list[IPhlowerModuleAdapter] = []
        for _setting in setting.modules:
            if isinstance(_setting, GroupModuleSetting):
                _layer = PhlowerGroupModule.from_setting(_setting)
                _modules.append(_layer)
                continue

            if isinstance(_setting, ModuleSetting):
                _layer = PhlowerModuleAdapter.from_setting(_setting)
                _modules.append(_layer)
                continue

            raise NotImplementedError()

        return PhlowerGroupModule(
            modules=_modules,
            name=setting.name,
            no_grad=setting.no_grad,
            input_keys=setting.get_input_keys(),
            output_keys=setting.get_output_keys(),
            destinations=setting.destinations,
        )

    def __init__(
        self,
        modules: list[IPhlowerModuleAdapter],
        name: str,
        no_grad: bool,
        input_keys: list[str],
        destinations: list[str],
        output_keys: list[str],
    ) -> None:
        super().__init__()
        self._phlower_modules = modules
        self._name = name
        self._no_grad = no_grad
        self._input_keys = input_keys
        self._destinations = destinations
        self._output_keys = output_keys

        self._stream = self.resolve()
        for _module in self._phlower_modules:
            self.add_module(_module.name, _module)

    @property
    def name(self) -> str:
        return self._name

    def get_display_info(self) -> str:
        return (
            f"input_keys: {self._input_keys}\noutput_keys: {self._output_keys}"
        )

    def get_n_nodes(self) -> list[int]:
        return None

    def resolve(self) -> dagstream.DagStream:
        for module in self._phlower_modules:
            module.resolve(parent=self)

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

        return stream

    def draw(
        self, output_directory: pathlib.Path | str, recursive: bool = True
    ):
        # TODO: Need to apply Dependency Injection. Use composition pattern.
        drawer = MermaidDrawer()

        if isinstance(output_directory, str):
            output_directory = pathlib.Path(output_directory)

        drawer.output(
            self._phlower_modules, output_directory / f"{self.name}.mmd"
        )

        if not recursive:
            return

        for module in self._phlower_modules:
            module.draw(output_directory, recursive=recursive)

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor] | None = None,
        **kwards,
    ) -> IPhlowerTensorCollections:
        results = phlower_tensor_collection({})

        # construct (and check dags)
        dag_modules = self._stream.construct()

        while dag_modules.is_active:
            nodes = dag_modules.get_ready()
            for node in nodes:
                if node.n_predecessors == 0:
                    inputs = phlower_tensor_collection(
                        {key: data[key] for key in self._input_keys}
                    )
                    node.receive_args(inputs)

                args = reduce_collections(node.get_received_args())
                _module: PhlowerModuleAdapter = node.get_user_function()
                _result = _module.forward(args, supports=supports, **kwards)

                dag_modules.send(node.mut_name, _result)
                dag_modules.done(node.mut_name)

                if dag_modules.check_last(node):
                    results.update(_result)

        return results

    def get_destinations(self) -> list[str]:
        return self._destinations

    def search_module(self, name: str) -> IPhlowerCoreModule:
        for _module in self._phlower_modules:
            if _module.name == name:
                return _module.get_core_module()

        raise KeyError(f"Module {name} is not found in group {self.name}.")

    def load_checkpoint_file(
        self,
        checkpoint_file: IPhlowerCheckpointFile,
        device: str | None = None,
        decrypt_key: bytes | None = None,
    ) -> None:
        content = checkpoint_file.load(device=device, decrypt_key=decrypt_key)

        key_name = TrainerSavedKeyType.MODEL_STATE_DICT.value
        if key_name not in content:
            raise KeyError(
                f"Key named {key_name} is not found in "
                f"{checkpoint_file.file_path}"
            )
        self.load_state_dict(
            content[TrainerSavedKeyType.MODEL_STATE_DICT.value]
        )

    def get_core_module(self) -> IPhlowerCoreModule:
        raise ValueError(
            "`get_core_module` cannot be called in PhlowerGroupModule."
        )
