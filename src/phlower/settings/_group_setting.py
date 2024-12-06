from __future__ import annotations

import functools
from typing import Literal

import pydantic
from dagstream.utils.errors import DagStreamCycleError
from pydantic import Field
from pydantic import dataclasses as dc
from typing_extensions import Self

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)
from phlower.settings._module_setting import ModuleSetting
from phlower.settings._nonlinear_solver_setting import SolverParameters
from phlower.settings._resolver import resolve_modules
from phlower.utils.exceptions import (
    PhlowerIterationSolverSettingError,
    PhlowerModuleCycleError,
    PhlowerModuleDuplicateKeyError,
    PhlowerModuleDuplicateNameError,
    PhlowerModuleKeyError,
    PhlowerModuleNodeDimSizeError,
)


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class GroupIOSetting:
    name: str
    n_last_dim: int


class GroupModuleSetting(
    pydantic.BaseModel, IModuleSetting, IReadOnlyReferenceGroupSetting
):
    name: str
    """
    name of group
    """

    inputs: list[GroupIOSetting] = Field(
        default_factory=list, validate_default=True
    )
    """
    definition of input varaibles
    """

    outputs: list[GroupIOSetting] = Field(
        default_factory=list, validate_default=True
    )
    """
    definition of output varaibles
    """

    modules: list[ModuleSetting | GroupModuleSetting] = Field(
        default_factory=list
    )
    """
    modules which belongs to this group
    """

    destinations: list[str] = Field(default_factory=list, frozen=True)
    """
    name of destination modules.
    """

    nn_type: Literal["Group", "GROUP"] = Field("Group", frozen=True)
    """
    name of neural network type. Fixed to "Group" or "GROUP"
    """

    no_grad: bool = False
    """
    A Flag not to calculate gradient. Defauls to False.
    """

    solver_type: str = "none"
    """
    Solver name calculating in iteration loop

    none: No iteration.
    simple: run iteration until calculated value is converged.
    bb: Barzilan-Borwein method is used to converge iteration results.
    """

    is_steady_problem: bool = False
    """
    When true, updating function in the group iteraion
      is defined as steady problems
    """

    solver_parameters: SolverParameters = Field(
        default_factory=dict, validate_default=True
    )
    """
    Parameters to pass iteration solver.  Contents depends on solver_type
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    @pydantic.field_validator("modules", mode="before")
    @classmethod
    def validate_annotate_modules(
        cls, vals: list
    ) -> list[ModuleSetting | GroupModuleSetting]:
        if not isinstance(vals, list):
            raise ValueError(f"'modules' expected to be List. actual: {vals}")

        parsed_items: list[ModuleSetting | GroupModuleSetting] = []
        for v in vals:
            if isinstance(v, dict):
                type_v = v.get("nn_type")
            else:
                type_v = getattr(v, "nn_type", None)

            if (type_v == "GROUP") or (type_v == "Group"):
                parsed_items.append(GroupModuleSetting(**v))
            else:
                parsed_items.append(ModuleSetting(**v))

        return parsed_items

    @pydantic.field_validator("inputs", "outputs")
    @classmethod
    def _check_unique_items(
        cls, values: list[GroupIOSetting], info: pydantic.ValidationInfo
    ) -> list[GroupIOSetting]:
        names = [v.name for v in values]
        if len(names) != len(set(names)):
            raise PhlowerModuleDuplicateKeyError(
                f"duplicate keys exist in {info.field_name}."
                f" GROUP = {info.data['name']}"
            )
        return values

    @pydantic.model_validator(mode="after")
    def _check_duplicate_module_name(self) -> Self:
        _names: set[str] = {v.name for v in self.modules}
        if len(_names) == len(self.modules):
            return self

        for module in self.modules:
            if module.get_name() in _names:
                raise PhlowerModuleDuplicateNameError(
                    f"Duplicate module name '{module.name}' is detected "
                    f"in {self.name}"
                )

        # it is not supposed to be reached here
        raise NotImplementedError()

    def get_name(self) -> str:
        return self.name

    def get_destinations(self) -> list[str]:
        return self.destinations

    def get_input_keys(self) -> list[str]:
        return [v.name for v in self.inputs]

    def get_output_keys(self) -> list[str]:
        return [v.name for v in self.outputs]

    def get_output_info(self) -> dict[str, int]:
        return {output.name: output.n_last_dim for output in self.outputs}

    def search_module_setting(self, name: str) -> IPhlowerLayerParameters:
        for module in self.modules:
            if module.name == name:
                return module.nn_parameters

        raise KeyError(
            f"ModuleSetting {name} is not found in GroupSetting {self.name}."
        )

    def resolve(
        self,
        *resolved_outputs: dict[str, int],
        is_first: bool = False,
        **kwards,
    ) -> None:
        if not is_first:
            self._check_inputs(*resolved_outputs)

        input_info = {v.name: v.n_last_dim for v in self.inputs}

        try:
            results = resolve_modules(input_info, self.modules, self)
        except DagStreamCycleError as ex:
            raise PhlowerModuleCycleError(
                f"A cycle is detected in {self.name}."
            ) from ex

        # check last state
        self._check_last_outputs(*results)

        # check target of solver setting if exist
        self._check_solver_target_exists_in_inputs_and_outputs()

    def find_module(
        self, name: str
    ) -> ModuleSetting | GroupModuleSetting | None:
        for v in self.modules:
            if v.name == name:
                return v

        return None

    def _check_inputs(self, *resolved_outputs: dict[str, int]) -> None:
        _flatten_dict = functools.reduce(lambda x, y: x | y, resolved_outputs)
        _n_keys = sum(len(v.keys()) for v in resolved_outputs)

        if len(_flatten_dict) != _n_keys:
            raise PhlowerModuleDuplicateKeyError(
                "Duplicate key name is detected in input keys "
                f"for {self.name}. Please check precedents."
            )

        if len(self.inputs) == 0:
            # set automatically
            self.inputs = [
                GroupIOSetting(name=k, n_last_dim=v)
                for k, v in _flatten_dict.items()
            ]

        for input_v in self.inputs:
            if input_v.name not in _flatten_dict:
                raise PhlowerModuleKeyError(
                    f"{input_v.name} is not passed to {self.name}. "
                    "Please check precedents."
                )

            if _flatten_dict[input_v.name] != input_v.n_last_dim:
                raise PhlowerModuleNodeDimSizeError(
                    f"n_dim of {input_v.name} in {self.name} "
                    f"is {input_v.n_last_dim}. "
                    "It is not consistent with the precedent modules"
                )

    def _check_last_outputs(self, *resolved_outputs: dict[str, int]) -> None:
        _flatten_dict = functools.reduce(lambda x, y: x | y, resolved_outputs)
        _n_keys = sum(len(v.keys()) for v in resolved_outputs)

        if len(_flatten_dict) != _n_keys:
            raise PhlowerModuleDuplicateKeyError(
                "Duplicate key name is detected in input keys "
                f"for {self.name}. Please check precedents"
            )

        if len(self.outputs) == 0:
            # set automatically
            self.outputs = [
                GroupIOSetting(name=k, n_last_dim=v)
                for k, v in _flatten_dict.items()
            ]

        for self_output in self.outputs:
            if self_output.name not in _flatten_dict:
                raise PhlowerModuleKeyError(
                    f"{self_output.name} is not created in {self.name}. "
                    "Please check last module in this group."
                )

            if _flatten_dict[self_output.name] != self_output.n_last_dim:
                raise PhlowerModuleNodeDimSizeError(
                    f"n_dim of {self_output.name} in {self.name} "
                    f"is {self_output.n_last_dim}. "
                    "It is not consistent with the precedent modules"
                )

    def _check_solver_target_exists_in_inputs_and_outputs(self) -> None:
        input_keys = self.get_input_keys()
        output_keys = self.get_output_keys()

        for key in self.solver_parameters.get_target_keys():
            if key not in input_keys:
                raise PhlowerIterationSolverSettingError(
                    f"{key} is missing in inputs."
                )
            if key not in output_keys:
                raise PhlowerIterationSolverSettingError(
                    f"{key} is missing in outputs."
                )

        return
