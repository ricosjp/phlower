from __future__ import annotations

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)


class RearrangeSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    pattern: str = Field(frozen=True)
    n_last_feature: int = Field(frozen=True)
    save_as_time_series: bool = Field(False, frozen=True)
    save_as_voxel: bool = Field(False, frozen=True)
    axes_lengths: dict[str, int] = Field(default_factory=dict, frozen=True)

    # This property only overwritten when resolving.
    nodes: list[int] | None = Field(None)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")

    @pydantic.field_validator("n_last_feature", mode="before")
    def check__valid_n_last_feature(cls, value: int) -> int:
        if value < 1:
            raise ValueError(
                f"n_last_feature must be positive value. Input: {value}"
            )
        return value

    def confirm(self, self_module: IModuleSetting) -> None:
        return

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) != 1:
            raise ValueError("only one input is allowed in Rearrange.")
        return input_dims[0]

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        n_dim = self.gather_input_dims(*input_dims)
        return [n_dim, self.n_last_feature]

    @pydantic.field_validator("nodes", mode="before")
    @classmethod
    def check_n_nodes(cls, vals: list[int] | None) -> list[int] | None:
        if vals is None:
            return vals

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

    def get_n_nodes(self) -> list[int] | None:
        return self.nodes

    def overwrite_nodes(self, nodes: list[int]) -> None:
        if len(nodes) != 2:
            raise ValueError(
                f"Invalid length of nodes to overwrite. Input: {nodes}"
            )

        if nodes[-1] != self.n_last_feature:
            raise ValueError(
                "the value of n_last_feature and "
                "the last item of nodes are not consistent."
            )
        if nodes[0] <= 0:
            raise ValueError("Resolved nodes must be positive.")
        self.nodes = nodes

    @property
    def need_reference(self) -> bool:
        return False

    def get_reference(self, parent: IReadOnlyReferenceGroupSetting):
        return
