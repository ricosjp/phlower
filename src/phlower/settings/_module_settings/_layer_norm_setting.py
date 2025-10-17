from __future__ import annotations

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)


class LayerNormSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    # This property only overwritten when resolving.
    nodes: list[int] | None = Field(None)
    eps: float = Field(1e-5, frozen=True)
    elementwise_affine: bool = Field(True, frozen=True)
    bias: bool = Field(True, frozen=True)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    def get_nn_type(cls) -> str:
        return "LayerNorm"

    def confirm(self, self_module: IModuleSetting) -> None:
        return

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) != 1:
            raise ValueError("only one input is allowed in LayerNorm.")
        return input_dims[0]

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        n_dim = self.gather_input_dims(*input_dims)
        return [n_dim, n_dim]

    @pydantic.field_validator("nodes", mode="before")
    @classmethod
    def check_n_nodes(cls, vals: list[int]) -> list[int]:
        if len(vals) != 2:
            raise ValueError(
                "size of nodes must be 2 in LayerNormSettings."
                f" input: {vals}"
            )

        if vals[1] <= 0:
            raise ValueError(
                "nodes in LayerNorm is inconsistent. "
                f"value {vals[1]} in nodes[1] is not allowed."
            )

        if vals[0] == -1:
            return vals

        if vals[0] != vals[1]:
            raise ValueError("Only same nodes are allowed in LayerNormSetting.")

        return vals

    def get_n_nodes(self) -> list[int] | None:
        return self.nodes

    def overwrite_nodes(self, nodes: list[int]) -> None:
        self.nodes = nodes

    @property
    def need_reference(self) -> bool:
        return False

    def get_reference(self, parent: IReadOnlyReferenceGroupSetting):
        return
