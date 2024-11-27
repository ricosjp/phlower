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

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) == 0:
            raise ValueError("zero input is not allowed in Reducer.")
        for i in range(1, len(input_dims)):
            if input_dims[0] != input_dims[i]:
                raise ValueError("input dimensions should be same in Reducer.")
        return input_dims[0]

    def confirm(self, self_module: IModuleSetting) -> None: ...

    @pydantic.field_validator("nodes")
    @classmethod
    def check_n_nodes(cls, vals: list[int]) -> list[int]:
        if vals is None:
            return vals

        """
        if len(vals) != 2:
            raise ValueError(
                "size of nodes must be 2 in ReducerSettings."
                f" input: {vals}"
            )
        """
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
