from __future__ import annotations

import pydantic
from pydantic import Field
from typing_extensions import Self

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)


class AccessorSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    # This property only overwritten when resolving.
    nodes: list[int] | None = Field(None)
    activation: str = Field("identity", frozen=True)
    index: list[int] | int = Field(...)
    dim: int = Field(0)
    keepdim: bool = Field(False)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")

    @classmethod
    def get_nn_type(cls) -> str:
        return "Accessor"

    def confirm(self, self_module: IModuleSetting) -> None:
        return

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) != 1:
            raise ValueError("only one input is allowed in Accessor.")
        return input_dims[0]

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        n_dim = self.gather_input_dims(*input_dims)
        if self.dim != -1:
            return [n_dim, n_dim]
        if not self.keepdim:
            raise ValueError(
                "access to the last dimension is invalid when keepdim=False"
            )
        new_size = 1
        if isinstance(self.index, list):
            new_size = len(self.index)
        return [n_dim, new_size]

    @pydantic.field_validator("nodes")
    @classmethod
    def check_n_nodes(cls, vals: list[int]) -> list[int]:
        return vals

    @pydantic.model_validator(mode="after")
    def check_keepdim(self) -> Self:
        if self.keepdim:
            return self

        if isinstance(self.index, list):
            if len(self.index) > 1:
                raise ValueError(
                    "multiple indices is invalid when keepdim=False"
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
