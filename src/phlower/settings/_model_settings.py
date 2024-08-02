from __future__ import annotations

import abc
from typing import Literal

import dagstream
import pydantic
from dagstream.utils.errors import DagStreamCycleError
from pydantic import Field, ValidationInfo
from pydantic import dataclasses as dc
from typing_extensions import Self

from phlower.settings._module_parameter_setting import PhlowerModuleParameters
from phlower.settings._module_settings import (
    gather_input_dims
)
from phlower.utils.exceptions import (
    PhlowerModuleCycleError,
    PhlowerModuleDuplicateKeyError,
    PhlowerModuleDuplicateNameError,
    PhlowerModuleKeyError,
    PhlowerModuleNodeDimSizeError,
)


class IModuleSetting(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_name(self) -> str: ...

    @abc.abstractmethod
    def get_destinations(self) -> list[str]: ...

    @abc.abstractmethod
    def get_output_info(self) -> dict[str, int]: ...

    @abc.abstractmethod
    def resolve(self, *resolved_outputs: dict[str, int]) -> None: ...


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class ModuleIOSetting:
    name: str
    n_dim: int


class GroupModuleSetting(IModuleSetting, pydantic.BaseModel):
    name: str
    inputs: list[ModuleIOSetting]
    outputs: list[ModuleIOSetting]
    modules: list[ModuleSetting | GroupModuleSetting]
    destinations: list[str] = Field(default_factory=lambda: [])
    nn_type: Literal["GROUP"] = "GROUP"
    no_grad: bool = False
    support_names: list[str] = pydantic.Field(default_factory=lambda: [])

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    @pydantic.field_validator("modules", mode="before")
    @classmethod
    def validate_annotate_modules(cls, vals):
        if not isinstance(vals, list):
            raise ValueError(f"'modules' expected to be List. actual: {vals}")

        parsed_items: list[ModuleSetting | GroupModuleSetting] = []
        for v in vals:
            if isinstance(v, dict):
                type_v = v.get("nn_type")
            else:
                type_v = getattr(v, "nn_type", None)

            if type_v == "GROUP":
                parsed_items.append(GroupModuleSetting(**v))
            else:
                parsed_items.append(ModuleSetting(**v))

        return parsed_items

    @pydantic.model_validator(mode="after")
    def _check_duplicate_name(self) -> Self:
        _names: set[str] = set()
        for module in self.modules:
            if module.get_name() in _names:
                raise PhlowerModuleDuplicateNameError(
                    f"Duplicate module name '{module.name}' is detected "
                    f"in {self.name}"
                )

        return self

    def get_name(self) -> str:
        return self.name

    def get_destinations(self) -> list[str]:
        return self.destinations

    def get_input_keys(self) -> list[str]:
        return [v.name for v in self.inputs]

    def get_output_keys(self) -> list[str]:
        return [v.name for v in self.outputs]

    def get_output_info(self) -> dict[str, int]:
        return {output.name: output.n_dim for output in self.outputs}

    def resolve(
        self, *resolved_outputs: dict[str, int], is_first: bool = False
    ) -> None:
        if not is_first:
            self._check_keys(*resolved_outputs)
            self._check_nodes(*resolved_outputs)

        input_info = {v.name: v.n_dim for v in self.inputs}

        try:
            results = _resolve_modules(input_info, self.modules)
        except DagStreamCycleError as ex:
            raise PhlowerModuleCycleError(
                f"A cycle is detected in {self.name}."
            ) from ex

        # check last state
        self._check_last_outputs(*results)

    def find_module(
        self, name: str
    ) -> ModuleSetting | GroupModuleSetting | None:
        for v in self.modules:
            if v.name == name:
                return v

        return None

    def _check_keys(self, *resolved_outputs: dict[str, int]) -> None:
        _to_input_keys: set[str] = set()
        n_keys: int = 0

        for output in resolved_outputs:
            _to_input_keys.update(output.keys())
            n_keys += len(output)

        if len(_to_input_keys) != n_keys:
            raise PhlowerModuleDuplicateKeyError(
                "Duplicate key name is detected in input keys "
                f"for {self.name}. Please check precedents."
            )

        for input_v in self.inputs:
            if input_v.name not in _to_input_keys:
                raise PhlowerModuleKeyError(
                    f"{input_v.name} is not passed to {self.name}. "
                    "Please check precedents."
                )

    def _check_nodes(self, *resolved_outputs: dict[str, int]) -> None:
        key2node: dict[str, int] = {}
        for output in resolved_outputs:
            # NOTE: Duplicate key error is check in self._check_keys
            key2node.update(output)

        for input_item in self.inputs:
            if key2node[input_item.name] != input_item.n_dim:
                raise PhlowerModuleNodeDimSizeError(
                    f"n_dim of {input_item.name} in {self.name} "
                    f"is {input_item.n_dim}. "
                    "It is not consistent with the precedent modules"
                )

    def _check_last_outputs(self, *resolved_outputs: dict[str, int]) -> None:
        key2node: dict[str, int] = {}
        for output in resolved_outputs:
            for key, val in output.items():
                if key in key2node:
                    raise PhlowerModuleDuplicateKeyError(
                        "Duplicate key name is detected in input keys "
                        f"for {self.name}. Please check precedents"
                    )
                key2node[key] = val

        for self_output in self.outputs:
            if self_output.name not in key2node:
                raise PhlowerModuleKeyError(
                    f"{self_output.name} is not created in {self.name}. "
                    "Please check last module in this group."
                )

            if key2node[self_output.name] != self_output.n_dim:
                raise PhlowerModuleNodeDimSizeError(
                    f"n_dim of {self_output.name} in {self.name} "
                    f"is {self_output.n_dim}. "
                    "It is not consistent with the precedent modules"
                )


class ModuleSetting(IModuleSetting, pydantic.BaseModel):
    nn_type: str = Field(..., frozen=True)
    name: str = Field(..., frozen=True)
    input_keys: list[str] = Field(..., frozen=True)
    output_key: str = Field(..., frozen=True)
    destinations: list[str] = Field(
        ..., default_factory=lambda: [], frozen=True
    )
    no_grad: bool = Field(False, frozen=True)
    nn_parameters: PhlowerModuleParameters = Field(
        default_factory=lambda: {}, validate_default=True
    )

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(
        extra="forbid", arbitrary_types_allowed=True
    )

    def get_name(self) -> str:
        return self.name

    def get_destinations(self) -> list[str]:
        return self.destinations

    def get_output_info(self) -> dict[str, int]:
        return {self.output_key: self.nn_parameters.get_n_nodes()[-1]}

    def resolve(self, *resolved_outputs: dict[str, int]) -> None:
        self._check_keys(*resolved_outputs)
        _resolved_nodes = self._resolve_nodes(*resolved_outputs)
        # NOTE: overwrite nodes
        self.nn_parameters.overwrite_nodes(_resolved_nodes)

    def _check_keys(self, *resolved_outputs: dict[str, int]) -> None:
        _to_input_keys: set[str] = set()
        n_keys: int = 0

        for output in resolved_outputs:
            _to_input_keys.update(output.keys())
            n_keys += len(output)

        if len(_to_input_keys) != n_keys:
            raise PhlowerModuleDuplicateKeyError(
                "Duplicate key name is detected in input keys for "
                f"{self.name}. Please check precedents"
            )

        for input_key in self.input_keys:
            if input_key not in _to_input_keys:
                raise PhlowerModuleKeyError(
                    f"{input_key} is not passed to {self.name}. "
                    "Please check precedents."
                )

    def _resolve_nodes(self, *resolved_outputs: dict[str, int]) -> list[int]:
        key2node: dict[str, int] = {}
        for output in resolved_outputs:
            key2node.update(output)

        try:
            first_node = gather_input_dims(
                self.nn_type, *(key2node[key] for key in self.input_keys)
            )
        except ValueError as ex:
            raise PhlowerModuleNodeDimSizeError(
                f"inputs for {self.name} have problems"
            ) from ex

        nodes = self.nn_parameters.get_n_nodes()
        if nodes is None:
            return [first_node, first_node]

        if (nodes[0] != -1) and (nodes[0] != first_node):
            raise PhlowerModuleNodeDimSizeError(
                f"nodes in {self.name} is {nodes}. "
                "It is not consistent with the precedent modules"
            )

        assert len(nodes) > 1
        if nodes[-1] == -1:
            raise PhlowerModuleNodeDimSizeError(
                "last number of nodes is must be "
                "integer number which is larger than 0."
                f" {self.name}"
            )

        new_nodes = [first_node]
        new_nodes.extend(nodes[1:])
        return new_nodes


class _SettingResolverAdapter:
    def __init__(self, setting: IModuleSetting) -> None:
        self._setting = setting

    @property
    def name(self) -> str:
        return self._setting.get_name()

    def __call__(self, *resolved_output: dict[str, int]) -> int:
        self._setting.resolve(*resolved_output)
        return self._setting.get_output_info()


def _resolve_modules(
    starts: dict[str, int], modules: list[IModuleSetting]
) -> list[dict[str, int]]:
    stream = dagstream.DagStream()
    resolvers = [_SettingResolverAdapter(layer) for layer in modules]
    name2node = {
        _resolver.name: stream.emplace(_resolver)[0] for _resolver in resolvers
    }

    for layer in modules:
        node = name2node[layer.get_name()]
        for to_name in layer.get_destinations():
            node.precede(name2node[to_name], pipe=True)

    _dag = stream.construct()

    executor = dagstream.StreamExecutor(_dag)
    results = executor.run(first_args=(starts,))

    return list(results.values())
