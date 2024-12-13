from __future__ import annotations

import pydantic
from pydantic import Field

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
    input_orders: list[str] = Field(default_factory=list)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")

    def gather_input_dims(self, *input_dims: int) -> int:
        assert len(input_dims) > 0
        sum_dim = sum(v for v in input_dims)
        return sum_dim

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        sum_dim = self.gather_input_dims(*input_dims)
        return (sum_dim, self.nodes[1])

    def confirm(self, self_module: IModuleSetting) -> None:
        input_keys = self_module.get_input_keys()
        if not self.input_orders:
            self.input_orders = input_keys

        if len(self.input_orders) != len(input_keys):
            raise ValueError(
                "the number of nodes isn't equal to that of input_orders."
            )

        for order in self.input_orders:
            if order not in input_keys:
                raise ValueError(f"{order} is not defined in input_keys")

    @pydantic.model_validator(mode="after")
    def check_n_nodes(self) -> list[int]:
        vals = self.nodes
        if len(vals) != 2:
            raise ValueError(f"length of nodes must be 2. input: {vals}.")

        for i, v in enumerate(vals):
            if v > 0:
                continue

            if (i == 0) and (v == -1):
                continue

            raise ValueError(
                "nodes in Rearrange is inconsistent. "
                f"value {v} in {i}-th of nodes is not allowed."
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
        if len(nodes) != 2:
            raise ValueError(
                f"Invalid length of nodes to overwrite. Input: {nodes}"
            )

        if nodes[1] != self.nodes[1]:
            raise ValueError(
                "the value of n_last_feature and "
                "the last item of nodes are not consistent."
            )
        if nodes[0] <= 0:
            raise ValueError("Resolved nodes must be positive.")

        self.nodes = nodes
