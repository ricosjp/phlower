from __future__ import annotations

import pathlib
from collections.abc import Callable
from functools import partial

import dagstream
import torch

from phlower._fields import ISimulationField
from phlower.collections.tensors import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
    reduce_stack,
    reduce_update,
)
from phlower.io._files import IPhlowerCheckpointFile
from phlower.nn._interface_iteration_solver import (
    IFIterationSolver,
    IOptimizeProblem,
)
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IPhlowerGroup,
    IPhlowerModuleAdapter,
    IReadonlyReferenceGroup,
)
from phlower.nn._iteration_solvers import EmptySolver, get_iteration_solver
from phlower.nn._phlower_module_adapter import PhlowerModuleAdapter
from phlower.nn._preset_group_module_adapter import (
    PhlowerPresetGroupModuleAdapter,
)
from phlower.services.drawers import MermaidDrawer
from phlower.settings._group_setting import GroupModuleSetting, ModuleSetting
from phlower.settings._preset_group_setting import PresetGroupModuleSetting
from phlower.utils.enums import TrainerSavedKeyType

TargetFunctionType = Callable[
    [IPhlowerTensorCollections], IPhlowerTensorCollections
]


class _GroupOptimizeProblem(IOptimizeProblem):
    def __init__(
        self,
        initials: IPhlowerTensorCollections,
        step_forward: TargetFunctionType,
        steady_mode: bool = False,
    ):
        self._step_forward = step_forward

        # Copy not to update during optimizing
        self._fixed_initials = phlower_tensor_collection(
            {k: initials[k] for k in initials.keys()}
        )
        self._steady_mode = steady_mode

    def step_forward(
        self, h: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        return self._step_forward(h)

    def gradient(
        self,
        h: IPhlowerTensorCollections,
        update_keys: list[str],
        operator_keys: list[str] | None = None,
    ) -> IPhlowerTensorCollections:
        if operator_keys:
            operator_value = self._step_forward(h).mask(operator_keys)
        else:
            operator_value = self._step_forward(h).mask(update_keys) - h.mask(
                update_keys
            )

        if self._steady_mode:
            # R(u) = - D[u] dt
            residuals = operator_value * -1.0
        else:
            # R(u) = v - u(t) - D[u] dt
            residuals = (
                h.mask(update_keys)
                - self._fixed_initials.mask(update_keys)
                - operator_value
            )

        return residuals


class PhlowerGroupModule(
    torch.nn.Module,
    IPhlowerModuleAdapter,
    IReadonlyReferenceGroup,
    IPhlowerGroup,
):
    @classmethod
    def from_setting(cls, setting: GroupModuleSetting) -> PhlowerGroupModule:
        _modules: list[IPhlowerModuleAdapter] = []
        for _setting in setting.modules:
            if isinstance(_setting, GroupModuleSetting):
                _layer = PhlowerGroupModule.from_setting(_setting)
                _modules.append(_layer)
                continue

            if isinstance(_setting, PresetGroupModuleSetting):
                _layer = PhlowerPresetGroupModuleAdapter.from_setting(_setting)
                _modules.append(_layer)
                continue

            if isinstance(_setting, ModuleSetting):
                _layer = PhlowerModuleAdapter.from_setting(_setting)
                _modules.append(_layer)
                continue

            raise NotImplementedError()

        solver_cls = get_iteration_solver(setting.solver_type)

        return PhlowerGroupModule(
            modules=_modules,
            name=setting.name,
            no_grad=setting.no_grad,
            input_keys=setting.get_input_keys(),
            output_keys=setting.get_output_keys(),
            destinations=setting.destinations,
            is_steady_problem=setting.is_steady_problem,
            iteration_solver=solver_cls.from_setting(setting.solver_parameters),
            time_series_length=setting.time_series_length,
        )

    def __init__(
        self,
        modules: list[IPhlowerModuleAdapter],
        name: str,
        no_grad: bool,
        input_keys: list[str],
        destinations: list[str],
        output_keys: list[str],
        is_steady_problem: bool = False,
        iteration_solver: IFIterationSolver | None = None,
        time_series_length: int | None = None,
        **kwards,
    ) -> None:
        super().__init__(**kwards)
        if iteration_solver is None:
            iteration_solver = EmptySolver()

        self._phlower_modules = modules
        self._name = name
        self._no_grad = no_grad
        self._input_keys = input_keys
        self._destinations = destinations
        self._output_keys = output_keys
        self._is_steady_problem = is_steady_problem
        self._iteration_solver = iteration_solver
        self._time_series_length = time_series_length or 0
        assert self._time_series_length >= -1

        self._stream = self.resolve()
        for _module in self._phlower_modules:
            self.add_module(_module.name, _module)

    @property
    def name(self) -> str:
        return self._name

    @property
    def do_time_series_iteration(self) -> bool:
        return self._time_series_length != 0

    def get_display_info(self) -> str:
        return (
            f"input_keys: {self._input_keys}\noutput_keys: {self._output_keys}"
        )

    def get_n_nodes(self) -> list[int] | None:
        return None

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> dagstream.DagStream:
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

        if not output_directory.exists():
            output_directory.mkdir(parents=True, exist_ok=True)

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
        field_data: ISimulationField,
        **kwards,
    ) -> IPhlowerTensorCollections:
        if self.do_time_series_iteration:
            return self._forward_time_series(
                data, field_data=field_data, **kwards
            )

        return self._forward(data, field_data=field_data, **kwards)

    def _forward_time_series(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField,
        **kwards,
    ) -> IPhlowerTensorCollections:
        results: list[IPhlowerTensorCollections] = []

        assert isinstance(self._time_series_length, int)
        _time_series_length = (
            self._time_series_length
            if self._time_series_length > 0
            else data.get_time_series_length()
        )

        for time_index in range(_time_series_length):
            inputs = data.snapshot(time_index).clone()

            if time_index != 0:
                last_result = results[-1].clone()
                inputs.update(last_result, overwrite=True)

            results.append(
                self._forward(inputs, field_data=field_data, **kwards)
            )

        return reduce_stack(results, to_time_series=True)

    def _forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField,
        **kwards,
    ) -> IPhlowerTensorCollections:
        step_forward = partial(
            self.step_forward, field_data=field_data, **kwards
        )
        problem = _GroupOptimizeProblem(
            initials=data,
            step_forward=step_forward,
            steady_mode=self._is_steady_problem,
        )

        self._iteration_solver.zero_residuals()

        return self._iteration_solver.run(initial_values=data, problem=problem)

    def step_forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField,
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

                args = reduce_update(node.get_received_args())
                _module: PhlowerModuleAdapter = node.get_user_function()
                _result = _module.forward(args, field_data=field_data, **kwards)

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
        map_location: str | dict | None = None,
        weights_only: bool = False,
        decrypt_key: bytes | None = None,
    ) -> None:
        content = checkpoint_file.load(
            map_location=map_location,
            weights_only=weights_only,
            decrypt_key=decrypt_key,
        )

        key_name = TrainerSavedKeyType.model_state_dict.value
        if key_name not in content:
            raise KeyError(
                f"Key named {key_name} is not found in "
                f"{checkpoint_file.file_path}"
            )
        self.load_state_dict(
            content[TrainerSavedKeyType.model_state_dict.value]
        )

    def get_core_module(self) -> IPhlowerCoreModule:
        return self
