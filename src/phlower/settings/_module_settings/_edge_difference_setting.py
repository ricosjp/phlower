from __future__ import annotations

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)


class EdgeDifferenceSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    support_name: str = Field(frozen=True)
    with_norm: bool = Field(False, frozen=True)

    # This property only overwritten when resolving.
    nodes: list[int] | None = Field(None)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")

    @classmethod
    def get_nn_type(cls) -> str:
        return "EdgeDifference"

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) != 1:
            raise ValueError(
                "num of input should be 1 in EdgeDifference. "
                f"input: {input_dims}"
            )
        return input_dims[0]

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        n_dim = self.gather_input_dims(*input_dims)
        out_dim = n_dim + 1 if self.with_norm else n_dim
        return [n_dim, out_dim]

    def confirm(self, self_module: IModuleSetting) -> None: ...

    @pydantic.field_validator("nodes", mode="before")
    @classmethod
    def check_n_nodes(cls, vals: list[int] | None) -> list[int] | None:
        if vals is None:
            return vals

        if len(vals) != 2:
            raise ValueError(
                "Size of nodes must be 2 in EdgeDifferenceSetting. "
                f"input: {vals}"
            )
        return vals

    @property
    def need_reference(self) -> bool:
        return False

    def get_reference(self, parent: IReadOnlyReferenceGroupSetting):
        return

    def get_n_nodes(self) -> list[int] | None:
        return self.nodes

    def overwrite_nodes(self, nodes: list[int]) -> None:
        self.nodes = nodes
