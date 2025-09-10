from __future__ import annotations

import pydantic
from pydantic import Field
from typing_extensions import Self

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)


class EinsumSetting(pydantic.BaseModel, IPhlowerLayerParameters):
    # This property only overwritten when resolving.
    nodes: list[int]
    equation: str = Field(frozen=True)
    activation: str = Field("identity", frozen=True)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")

    @classmethod
    def get_nn_type(cls) -> str:
        return "Einsum"

    def gather_input_dims(self, *input_dims: int) -> int:
        assert len(input_dims) > 0
        sum_dim = sum(v for v in input_dims)
        return sum_dim

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        sum_dim = self.gather_input_dims(*input_dims)
        return (sum_dim, self.nodes[1])

    def confirm(self, self_module: IModuleSetting) -> None: ...

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
                "nodes in Einsum is inconsistent. "
                f"value {v} in {i}-th of nodes is not allowed."
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
        if len(nodes) != 2:
            raise ValueError(
                f"Invalid length of nodes to overwrite. Input: {nodes}"
            )

        if nodes[1] != self.nodes[1]:
            raise ValueError("the last value of nodes is not consistent.")
        if nodes[0] <= 0:
            raise ValueError("Resolved nodes must be positive.")

        self.nodes = nodes
