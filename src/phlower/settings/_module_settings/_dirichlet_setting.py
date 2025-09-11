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
    dirichlet_name: str

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")

    @classmethod
    def get_nn_type(cls) -> str:
        return "Dirichlet"

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) != 2:
            raise ValueError("num of input should be 2 in Dirichlet.")
        sum_dim = sum(v for v in input_dims)
        return sum_dim

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        n_dim = self.gather_input_dims(*input_dims)
        out_dim = max(input_dims)
        return [n_dim, out_dim]

    def confirm(self, self_module: IModuleSetting) -> None:
        input_keys = self_module.get_input_keys()
        if self.dirichlet_name not in input_keys:
            raise ValueError(
                f"{self.dirichlet_name} is not found in input_keys. "
                "Please check predecessors."
            )

    @pydantic.field_validator("nodes")
    @classmethod
    def check_n_nodes(cls, vals: list[int]) -> list[int]:
        if vals is None:
            return vals

        if len(vals) != 2:
            raise ValueError(
                "Size of nodes must be 2 in DirichletSettings."
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
