from __future__ import annotations

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)


class DirichletSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    # This property only overwritten when resolving.
    nodes: list[int] | None = Field(None)
    activation: str = Field("identity", frozen=True)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")

    def gather_input_dims(self, *input_dims: int) -> int:
        assert len(input_dims) > 0
        sum_dim = sum(v for v in input_dims)
        return sum_dim

    def confirm(self, self_module: IModuleSetting) -> None: ...

    @pydantic.field_validator("nodes")
    @classmethod
    def check_n_nodes(cls, vals: list[int]) -> list[int]:
        if vals is None:
            return vals

        if len(vals) != 2:
            raise ValueError(
                "size of nodes must be 2 in DirichletSettings."
                f" input: {vals}"
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
