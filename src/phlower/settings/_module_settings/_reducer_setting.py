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
        dims_not_one = [d for d in input_dims if d != 1]
        for i in range(1, len(dims_not_one)):
            if dims_not_one[0] != dims_not_one[i]:
                raise ValueError(
                    "input dimensions should be same in Reducer. "
                    f"(given: {input_dims})"
                )
        sum_dim = sum(v for v in input_dims)
        return sum_dim

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        sum_dim = self.gather_input_dims(*input_dims)
        dims_not_one = {d for d in input_dims if d != 1}
        if len(dims_not_one) == 0:
            output_dim = 1
        else:
            assert len(dims_not_one) == 1
            output_dim = dims_not_one.pop()
        return [sum_dim, output_dim]

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
