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
    output_feature_dim: int = Field(frozen=True)
    save_as_time_series: bool = Field(False, frozen=True)
    save_as_voxel: bool = Field(False, frozen=True)
    axes_lengths: dict[str, int] = Field(default_factory=dict, frozen=True)

    # This property only overwritten when resolving.
    nodes: list[int] | None = Field(None)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")

    @classmethod
    def get_nn_type(cls) -> str:
        return "Rearrange"

    @pydantic.field_validator("output_feature_dim", mode="before")
    def check__valid_output_feature_dim(cls, value: int) -> int:
        if value < 1:
            raise ValueError(
                "output_feature_dim must be positive value. " f"Input: {value}"
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
        return [n_dim, self.output_feature_dim]

    @pydantic.field_validator("nodes", mode="before")
    @classmethod
    def check_n_nodes(cls, vals: list[int] | None) -> list[int] | None:
        if vals is None:
            return vals

        if len(vals) != 2:
            raise ValueError(f"length of nodes must be 2. input: {vals}")

        for i, v in enumerate(vals):
            if v > 0:
                continue

            if (i == 0) and (v == -1):
                continue

            raise ValueError(
                "nodes in Rearrange is inconsistent. "
                f"value {v} is not allowed except the head node."
            )

        return vals

    def get_n_nodes(self) -> list[int] | None:
        return self.nodes

    def overwrite_nodes(self, nodes: list[int]) -> None:
        if len(nodes) != 2:
            raise ValueError(
                f"Invalid length of nodes to overwrite. Input: {nodes}"
            )

        if nodes[-1] != self.output_feature_dim:
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
