from __future__ import annotations

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)


class ReducerSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    # This property only overwritten when resolving.
    nodes: list[int] | None = Field(None)
    activation: str = Field("identity", frozen=True)
    operator: str = Field(...)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")

    @classmethod
    def get_nn_type(cls) -> str:
        return "Reducer"

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) == 0:
            raise ValueError("zero input is not allowed in Reducer.")
        for i in range(1, len(input_dims)):
            if input_dims[0] != input_dims[i]:
                raise ValueError("input dimensions should be same in Reducer.")
        sum_dim = sum(v for v in input_dims)
        return sum_dim

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        sum_dim = self.gather_input_dims(*input_dims)
        if sum_dim % len(input_dims) != 0:
            raise ValueError(
                f"sum_dim({sum_dim}) is not divisible "
                f"by len(input_dims)({len(input_dims)})."
            )
        return [sum_dim, sum_dim // len(input_dims)]

    def confirm(self, self_module: IModuleSetting) -> None: ...

    @pydantic.field_validator("nodes")
    @classmethod
    def check_n_nodes(cls, vals: list[int]) -> list[int]:
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
