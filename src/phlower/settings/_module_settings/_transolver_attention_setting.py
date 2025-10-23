from __future__ import annotations

import pydantic
from pydantic import Field
from typing_extensions import Self

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)


class TransolverAttentionSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    # This property only overwritten when resolving.
    nodes: list[int]
    unbatch_key: str | None = Field(None, frozen=True)
    heads: int = Field(8, frozen=True)
    slice_num: int = Field(32, frozen=True)
    dropout: float = Field(0.0, frozen=True)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")

    @classmethod
    def get_nn_type(cls) -> str:
        return "TransolverAttention"

    def confirm(self, self_module: IModuleSetting) -> None:
        return

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) != 1:
            raise ValueError(
                "only one input is allowed in TransolverAttention."
            )
        return input_dims[0]

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        n_dim = self.gather_input_dims(*input_dims)
        return [n_dim, n_dim, n_dim]

    @pydantic.field_validator("nodes", mode="before")
    @classmethod
    def check_n_nodes(cls, vals: list[int]) -> list[int]:
        if len(vals) != 3:
            raise ValueError(
                "size of nodes must 3 in TransolverAttentionSetting."
                f" input: {vals}"
            )

        for i, v in enumerate(vals):
            if v > 0:
                continue

            if (i == 0) and (v == -1):
                continue

            raise ValueError(
                "nodes in TransolverAttention is inconsistent. "
                f"value {v} in {i}-th of nodes is not allowed."
            )

        return vals

    @pydantic.field_validator("heads", mode="before")
    @classmethod
    def check_heads(cls, val: int) -> int:
        if val <= 0:
            raise ValueError(
                f"heads must be positive in TransolverAttentionSetting. "
                f"input: {val}"
            )
        return val

    @pydantic.field_validator("slice_num", mode="before")
    @classmethod
    def check_slice_num(cls, val: int) -> int:
        if val <= 0:
            raise ValueError(
                f"slice_num must be positive in TransolverAttentionSetting. "
                f"input: {val}"
            )
        return val

    @pydantic.field_validator("dropout", mode="before")
    @classmethod
    def check_dropout(cls, val: float) -> float:
        if not (0.0 <= val < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1) in TransolverAttentionSetting. "
                f"input: {val}"
            )
        return val

    @pydantic.model_validator(mode="after")
    def check_heads_divisibility(self) -> Self:
        if self.nodes[1] % self.heads != 0:
            raise ValueError(
                f"nodes[1] ({self.nodes[1]}) must be divisible"
                f"by heads ({self.heads}) in TransolverAttentionSetting."
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
        if len(nodes) != 3:
            raise ValueError(
                f"Invalid length of nodes to overwrite. Input: {nodes}"
            )

        if nodes[0] <= 0 or nodes[1] <= 0 or nodes[2] <= 0:
            raise ValueError("Resolved nodes must be positive.")

        if nodes[1] % self.heads != 0:
            raise ValueError(
                f"nodes[1] ({nodes[1]}) must be divisible"
                f"by heads ({self.heads})"
            )

        self.nodes = nodes
