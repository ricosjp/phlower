from __future__ import annotations

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IReadOnlyReferenceGroupSetting,
)
from phlower.settings._module_parameter_setting import PhlowerModuleParameters
from phlower.utils.exceptions import (
    PhlowerModuleDuplicateKeyError,
    PhlowerModuleKeyError,
    PhlowerModuleNodeDimSizeError,
)


class ModuleSetting(IModuleSetting, pydantic.BaseModel):
    nn_type: str = Field(..., frozen=True)
    """
    name of neural network type. For example, GCN, MLP.
    """

    name: str = Field(..., frozen=True)
    """
    name of group
    """

    input_keys: list[str] = Field(..., frozen=True)
    """
    key names of input variables
    """

    output_key: str = Field(..., frozen=True)
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

    nn_parameters: PhlowerModuleParameters = Field(
        default_factory=dict, validate_default=True
    )
    """
    parameters for neural networks.
    Allowed items depend on `nn_type`.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(
        extra="forbid", arbitrary_types_allowed=True
    )

    def get_name(self) -> str:
        return self.name

    def get_input_keys(self) -> list[str]:
        return self.input_keys

    def get_destinations(self) -> list[str]:
        return self.destinations

    def get_output_info(self) -> dict[str, int]:
        return {self.output_key: self.nn_parameters.get_n_nodes()[-1]}

    def resolve(
        self,
        *resolved_outputs: dict[str, int],
        parent: IReadOnlyReferenceGroupSetting | None = None,
    ) -> None:
        self._check_keys(*resolved_outputs)

        if self.nn_parameters.need_reference:
            self.nn_parameters.get_reference(parent)

        _resolved_nodes = self._resolve_nodes(*resolved_outputs)
        # NOTE: overwrite nodes
        self.nn_parameters.overwrite_nodes(_resolved_nodes)
        self.nn_parameters.confirm(self)

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
            first_node = self.nn_parameters.gather_input_dims(
                *(key2node[key] for key in self.input_keys)
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
