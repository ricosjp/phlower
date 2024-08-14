from __future__ import annotations

import pydantic
from pydantic import Field
from typing_extensions import Self

from phlower.settings._interface import (
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)


class MLPSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    nodes: list[int] = Field(
        ...
    )  # This property only overwritten when resolving.
    activations: list[str] = Field(default_factory=lambda: [], frozen=True)
    dropouts: list[float] = Field(default_factory=lambda: [], frozen=True)
    bias: bool = Field(True, frozen=True)

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) != 1:
            raise ValueError("only one input is allowed in GCN.")
        return input_dims[0]

    @pydantic.field_validator("nodes")
    @classmethod
    def check_n_nodes(cls, vals: list[int]) -> list[int]:
        if len(vals) < 2:
            raise ValueError(
                "size of nodes must be larger than 1 in GCNSettings."
                f" input: {vals}"
            )

        for i, v in enumerate(vals):
            if v > 0:
                continue

            if (i == 0) and (v == -1):
                continue

            raise ValueError(
                "nodes in GCN is inconsistent. "
                f"value {v} in {i}-th of nodes is not allowed."
            )

        return vals

    @pydantic.model_validator(mode="after")
    def check_nodes_size(self) -> Self:
        if len(self.nodes) - 1 != len(self.activations):
            raise ValueError(
                "Size of nodes and activations is not compatible "
                "in GCNSettings."
                " len(nodes) must be equal to 1 + len(activations)."
            )
        return self

    def get_n_nodes(self) -> list[int] | None:
        return self.nodes

    def overwrite_nodes(self, nodes: list[int]) -> None:
        self.nodes = nodes

    @property
    def need_reference(self) -> bool:
        return False

    def get_reference(self, parent: IReadOnlyReferenceGroupSetting):
        return
