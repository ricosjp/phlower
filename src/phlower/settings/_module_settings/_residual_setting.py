from __future__ import annotations

from typing import Self

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)
from phlower.utils import parse_equation


class ResidualSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    # This property only overwritten when resolving.
    nodes: list[int]
    symbols_from_input: list[str] = Field(default_factory=list, frozen=True)
    """
    symbols_from_input is a list of symbols corresponding to the input data.
    """

    symbols_from_field: list[str] = Field(default_factory=list, frozen=True)
    """
    symbols_from_field is a list of symbols corresponding to the field data.
    """

    equation: str = Field(..., frozen=True)
    """
    equation is a string representing the equation to compute the residual.
    The equation can contain symbols from both input and field data.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    def get_nn_type(cls) -> str:
        return "Residual"

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) == 0:
            raise ValueError("zero input is not allowed in Reducer.")
        sum_dim = sum(v for v in input_dims)
        return sum_dim

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        sum_dim = self.gather_input_dims(*input_dims)
        return [sum_dim, self.nodes[1]]

    def confirm(self, self_module: IModuleSetting) -> None:
        # check symbols_from_input are included in self_module.input_names

        actual_inputs = set(self_module.get_input_keys())
        if not set(self.symbols_from_input).issubset(actual_inputs):
            raise ValueError(
                f"symbols_from_input {self.symbols_from_input} are not "
                f"subset of actual inputs {actual_inputs} in ResidualSetting."
            )
        return

    @pydantic.model_validator(mode="after")
    def try_to_parse_equation(self) -> Self:
        try:
            _ = parse_equation(
                self.equation,
                self.symbols_from_input + self.symbols_from_field,
            )
        except Exception as ex:
            raise ValueError(
                f"Failed to parse equation {self.equation} in ResidualSetting."
            ) from ex

        return self

    @pydantic.model_validator(mode="after")
    def check_n_nodes(self) -> Self:
        vals = self.nodes
        if len(vals) != 2:
            raise ValueError(f"length of nodes must be 2. input: {vals}.")

        for i, v in enumerate(vals):
            if v > 0:
                continue

            if (i == 0) and (v == -1):
                continue

            raise ValueError(
                "nodes in Residual is inconsistent. "
                f"value {v} in {i}-th of nodes is not allowed."
            )

        return self

    @pydantic.model_validator(mode="after")
    def check_unique_symbols(self) -> Self:
        all_symbols = self.symbols_from_input + self.symbols_from_field
        if len(all_symbols) != len(set(all_symbols)):
            raise ValueError(
                f"symbols_from_input {self.symbols_from_input} and "
                f"symbols_from_field {self.symbols_from_field} must be unique."
            )
        return self

    @property
    def need_reference(self) -> bool:
        return False

    def get_reference(self, parent: IReadOnlyReferenceGroupSetting):
        return

    def get_n_nodes(self) -> list[int] | None:
        return self.nodes

    def overwrite_nodes(self, nodes: list[int]) -> None:
        self.nodes = nodes
