from __future__ import annotations

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
    skip_converge: bool = False  # HACK need to fix


class GroupModuleSetting(
    pydantic.BaseModel, IModuleSetting, IReadOnlyReferenceGroupSetting
):
    name: str
    """
    name of group
    """

    inputs: list[GroupIOSetting]
    """
    definition of input varaibles
    """

    outputs: list[GroupIOSetting]
    """
    definition of output varaibles
    """

    modules: list[ModuleSetting | GroupModuleSetting]
    """
    modules which belongs to this group
    """

    destinations: list[str] = Field(default_factory=lambda: [])
    """
    name of destination modules.
    """

    nn_type: Literal["Group", "GROUP"] = "Group"
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

    solver_parameters: SolverParameters = pydantic.Field(
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

    @pydantic.model_validator(mode="after")
    def check_solver_target_exists_in_inputs_and_outputs(self) -> Self:
        input_keys = self.get_input_keys()
        output_keys = self.get_output_keys()

        for key in self.solver_parameters.get_target_keys():
            if key not in input_keys:
                raise ValueError(f"{key} is missing in inputs.")
            if key not in output_keys:
                raise ValueError(f"{key} is missing in outputs.")

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
            self._check_keys(*resolved_outputs)
            self._check_nodes(*resolved_outputs)

        input_info = {v.name: v.n_last_dim for v in self.inputs}

        try:
            results = resolve_modules(input_info, self.modules, self)
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
            if key2node[input_item.name] != input_item.n_last_dim:
                raise PhlowerModuleNodeDimSizeError(
                    f"n_dim of {input_item.name} in {self.name} "
                    f"is {input_item.n_last_dim}. "
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

            if key2node[self_output.name] != self_output.n_last_dim:
                raise PhlowerModuleNodeDimSizeError(
                    f"n_dim of {self_output.name} in {self.name} "
                    f"is {self_output.n_last_dim}. "
                    "It is not consistent with the precedent modules"
                )
