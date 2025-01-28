from __future__ import annotations

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)


class AccessorSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    # This property only overwritten when resolving.
    nodes: list[int] | None = Field(None)
    activation: str = Field("identity", frozen=True)
    index: int = Field(...)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")

    def confirm(self, self_module: IModuleSetting) -> None:
        return

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) != 1:
            raise ValueError("only one input is allowed in Accessor.")
        return input_dims[0]

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        n_dim = self.gather_input_dims(*input_dims)
        return [n_dim, n_dim]

    @pydantic.field_validator("nodes")
    @classmethod
    def check_n_nodes(cls, vals: list[int]) -> list[int]:
        if vals is None:
            return vals

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
