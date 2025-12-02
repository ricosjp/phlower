from __future__ import annotations

import functools
from typing import Literal

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IReadOnlyReferenceGroupSetting,
)
from phlower.settings._preset_group_parameter_setting import (
    PhlowerPresetGroupParameters,
)
from phlower.utils.exceptions import (
    PhlowerModuleDuplicateKeyError,
    PhlowerModuleKeyError,
    PhlowerModuleNodeDimSizeError,
)


class _PresetGroupNodeSetting(pydantic.BaseModel):
    name: str
    n_last_dim: int | None = None

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class PresetGroupModuleSetting(pydantic.BaseModel, IModuleSetting):
    nn_type: Literal[
        "PresetGroup",
        "PresetGROUP",
        "PRESETGROUP",
    ] = Field("PresetGroup", frozen=True)
    """
    name of neural network type.
    """

    name: str = Field(..., frozen=True)
    """
    name of group
    """

    preset_type: str = Field(..., frozen=True)
    """
    name of preset group. For example, Identity
    """

    inputs: list[_PresetGroupNodeSetting] = Field(
        default_factory=dict, frozen=False
    )
    """
    key names and dims of input variables
    """

    outputs: list[_PresetGroupNodeSetting] = Field(..., frozen=True)
    """
    key names and dims of output variables
    """

    destinations: list[str] = Field(
        ..., default_factory=lambda: [], frozen=True
    )
    """
    name of destination modules
    """

    no_grad: bool = Field(False, frozen=True)
    """
    A Flag not to calculate gradient. Default is False.
    """
    nn_parameters_same_as: str | None = Field(None)
    """
    name of another module to copy nn_parameters.
    """

    nn_parameters: PhlowerPresetGroupParameters = Field(
        default_factory=dict, validate_default=True
    )
    """
    parameters for neural networks.
    Allowed items depend on `preset_type`
    """
    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

    @pydantic.field_validator("inputs", "outputs")
    @classmethod
    def _check_unique_items(
        cls,
        values: list[_PresetGroupNodeSetting],
        info: pydantic.ValidationInfo,
    ) -> list[_PresetGroupNodeSetting]:
        names = [v.name for v in values]
        if len(names) != len(set(names)):
            raise PhlowerModuleDuplicateKeyError(
                f"duplicate keys exist in {info.field_name}."
                f" PresetGroup = {info.data['name']}"
            )
        return values

    def get_name(self) -> str:
        return self.name

    def get_preset_type(self) -> str:
        return self.preset_type

    def get_input_keys(self) -> list[str]:
        return [input_node.name for input_node in self.inputs]

    def get_output_keys(self) -> list[str]:
        return [output_node.name for output_node in self.outputs]

    def get_output_info(self) -> dict[str, int]:
        return {
            output_node.name: output_node.n_last_dim
            for output_node in self.outputs
        }

    def get_destinations(self) -> list[str]:
        return self.destinations

    def resolve(
        self,
        *resolved_outputs: dict[str, int],
        parent: IReadOnlyReferenceGroupSetting | None = None,
    ) -> None:
        try:
            if self.nn_parameters_same_as is not None:
                raise NotImplementedError(
                    "PresetGroupModuleSetting does not support "
                    "`nn_parameters_same_as` item."
                )

            _resolved_inputs = self._resolve_input_nodes(*resolved_outputs)
            # NOTE: overwrite input_nodes
            self._overwrite_inputs(_resolved_inputs)

            self.nn_parameters.confirm(self)
        except (
            KeyError,
            PhlowerModuleDuplicateKeyError,
            PhlowerModuleKeyError,
            PhlowerModuleNodeDimSizeError,
        ) as ex:
            raise ex
        except ValueError as ex:
            raise ValueError(
                "Resolving relationship between precedents failed "
                f"at {self.name}. \n"
                f"Error Details: {str(ex)}"
            ) from ex

    def _resolve_input_nodes(self, *resolved_outputs: dict[str, int]) -> None:
        _flatten_dict = functools.reduce(lambda x, y: x | y, resolved_outputs)
        _n_keys = sum(len(v.keys()) for v in resolved_outputs)

        if len(_flatten_dict) != _n_keys:
            raise PhlowerModuleDuplicateKeyError(
                "Duplicate key name is detected in input keys for "
                f"{self.name}. Please check precedents"
            )

        if len(self.inputs) == 0:
            # set automatically
            return _flatten_dict

        for input_v in self.inputs:
            if input_v.name not in _flatten_dict:
                raise PhlowerModuleKeyError(
                    f"{input_v.name} is not passed to {self.name}. "
                    "Please check precedents."
                )

            if input_v.n_last_dim is None:
                continue

            if input_v.n_last_dim == -1:
                continue

            if _flatten_dict[input_v.name] != input_v.n_last_dim:
                raise PhlowerModuleNodeDimSizeError(
                    f"n_last_dim of {input_v.name} in {self.name} "
                    f"is {input_v.n_last_dim}. "
                    "It is not consistent with the precedent modules"
                )
        return {
            input_v.name: _flatten_dict[input_v.name] for input_v in self.inputs
        }

    def _overwrite_inputs(self, input_nodes: dict[str, int]) -> None:
        self.inputs = [
            _PresetGroupNodeSetting(name=key, n_last_dim=dim)
            for key, dim in input_nodes.items()
        ]
