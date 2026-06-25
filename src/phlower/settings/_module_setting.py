from __future__ import annotations

import functools
import typing

import pydantic
from pydantic import Field

from phlower.settings._debug_parameter_setting import (
    PhlowerModuleDebugParameters,
)
from phlower.settings._interface import (
    IModuleSetting,
    IReadOnlyReferenceGroupSetting,
)
from phlower.settings._module_parameter_setting import (
    PhlowerModuleParameters,
)
from phlower.utils.exceptions import (
    PhlowerModuleDuplicateKeyError,
    PhlowerModuleKeyError,
    PhlowerModuleNodeDimSizeError,
)


class ModuleSetting(IModuleSetting, pydantic.BaseModel):
    nn_type: str = Field(frozen=True)
    """
    name of neural network type. For example, GCN, MLP.
    """

    name: str = Field(frozen=True)
    """
    name of group
    """

    input_keys: list[str] = Field(default_factory=list)
    """
    key names of input variables
    """

    input_keys_promoting_to_field: list[str] = Field(default_factory=list)
    """
    key names of input variables that will be promoted to field_data.
    """

    output_key: str = Field(default="")
    """
    key names of output variables
    """

    destinations: list[str] = Field(
        ..., default_factory=lambda: [], frozen=True
    )
    """
    name of destination modules.
    """

    no_grad: bool = Field(False, frozen=True)
    """
    A Flag not to calculate gradient. Defauls to False.
    """

    nn_parameters_same_as: str | None = Field(None)
    """
    name of another module to copy nn_parameters.
    """

    nn_parameters: PhlowerModuleParameters = Field(
        default_factory=dict, validate_default=True
    )
    """
    parameters for neural networks.
    Allowed items depend on `nn_type`.
    """

    coeff: float = Field(1.0, frozen=True)
    """
    coefficient to be applied to the module.
    """

    debug_parameters: PhlowerModuleDebugParameters = Field(
        default_factory=dict, validate_default=True
    )
    """
    parameters to debug networks.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, validate_assignment=True
    )

    def get_name(self) -> str:
        return self.name

    def get_input_keys(self) -> list[str]:
        return self.input_keys

    def get_destinations(self) -> list[str]:
        return self.destinations

    def get_output_info(self) -> dict[str, int]:
        return {self.output_key: self.nn_parameters.get_n_nodes()[-1]}

    def _get_actual_input_dims(
        self, key2node: dict[str, int]
    ) -> dict[str, int]:
        return {
            key: key2node[key]
            for key in self.input_keys
            if key not in self.input_keys_promoting_to_field
        }

    def resolve(
        self,
        *resolved_outputs: dict[str, int],
        parent: IReadOnlyReferenceGroupSetting | None = None,
    ) -> None:
        try:
            self._check_keys(*resolved_outputs)

            if self.nn_parameters_same_as is not None:
                ref = parent.search_module_setting(self.nn_parameters_same_as)
                assert isinstance(ref, pydantic.BaseModel)
                self.nn_parameters_same_as = None
                self.nn_parameters = ref.model_dump()

            if self.nn_parameters.need_reference:
                self.nn_parameters.get_reference(parent)

            _resolved_nodes = self._resolve_nodes(*resolved_outputs)
            # NOTE: overwrite nodes
            self.nn_parameters.overwrite_nodes(_resolved_nodes)

            if len(self.output_key) == 0:
                self.output_key = f"OUT_{self.name}"
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

    def _check_keys(self, *resolved_outputs: dict[str, int]) -> None:
        _flatten_dict = functools.reduce(lambda x, y: x | y, resolved_outputs)
        _n_keys = sum(len(v.keys()) for v in resolved_outputs)

        if len(_flatten_dict) != _n_keys:
            raise PhlowerModuleDuplicateKeyError(
                "Duplicate key name is detected in input keys for "
                f"{self.name}. Please check precedents"
            )

        if len(self.input_keys) == 0:
            # set automatically
            self.input_keys = list(_flatten_dict.keys())

        for input_key in self.input_keys:
            if input_key not in _flatten_dict:
                raise PhlowerModuleKeyError(
                    f"{input_key} is not passed to {self.name}. "
                    "Please check precedents."
                )

        for promotion_key in self.input_keys_promoting_to_field:
            if promotion_key not in self.input_keys:
                raise PhlowerModuleKeyError(
                    f"{promotion_key} is not found in input_keys of "
                    f"{self.name}."
                    "Please check the configuration of "
                    "input_keys_promoting_to_field."
                )

    def _resolve_nodes(self, *resolved_outputs: dict[str, int]) -> list[int]:
        key2node: dict[str, int] = {}
        for output in resolved_outputs:
            key2node.update(output)

        _hints = typing.get_type_hints(self.nn_parameters.gather_input_dims)
        if _hints["input_dims"] == dict[str, int]:
            first_node = self.nn_parameters.gather_input_dims(
                self._get_actual_input_dims(key2node)
            )
        else:
            # HACK: for backward compatibility.
            # We need to remove this branch
            # after changing all gather_input_dims to accept dict[str, int]
            first_node = self.nn_parameters.gather_input_dims(
                *list(self._get_actual_input_dims(key2node).values())
            )

        nodes = self.nn_parameters.get_n_nodes()
        if nodes is None:
            _hints = typing.get_type_hints(self.nn_parameters.get_default_nodes)
            if _hints["input_dims"] == dict[str, int]:
                return self.nn_parameters.get_default_nodes(
                    self._get_actual_input_dims(key2node)
                )
            else:
                # HACK: for backward compatibility.
                return self.nn_parameters.get_default_nodes(
                    *list(self._get_actual_input_dims(key2node).values())
                )

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
