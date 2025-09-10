from __future__ import annotations

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)


class SPMMSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    support_name: str
    factor: float = Field(1.0, frozen=True)
    transpose: bool = Field(False, frozen=True)

    nodes: list[int] | None = Field(None)
    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    def get_nn_type(cls) -> str:
        return "SPMM"

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) != 1:
            raise ValueError(
                f"Invalid the number of inputs to SPMM. input: {input_dims}"
            )
        return input_dims[0]

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        sum_dim = self.gather_input_dims(*input_dims)
        return (sum_dim, sum_dim)

    def confirm(self, self_module: IModuleSetting) -> None: ...

    @pydantic.field_validator("nodes", mode="before")
    @classmethod
    def check_n_nodes(cls, vals: list[int] | None) -> list[int]:
        if vals is None:
            return vals

        if len(vals) != 2:
            raise ValueError(f"length of nodes must be 2. input: {vals}.")

        for i, v in enumerate(vals):
            if v > 0:
                continue

            if (i == 0) and (v == -1):
                continue

            raise ValueError(
                "Nodes in SPMM is inconsistent. "
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

        if nodes[0] <= 0 and nodes[1] <= 0:
            raise ValueError("Resolved nodes must be positive.")

        self.nodes = nodes
