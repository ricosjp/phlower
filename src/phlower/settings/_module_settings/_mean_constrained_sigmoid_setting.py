from __future__ import annotations

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)


class MeanConstrainedSigmoidSetting(
    pydantic.BaseModel, IPhlowerLayerParameters
):
    # This property only overwritten when resolving.
    nodes: list[int] | None = Field(None)

    output_feature_dim: int = Field(frozen=True)
    reference_variable_name: str = Field(frozen=True)
    weight_field_name: str | None = Field(None, frozen=True)
    slope: float = Field(2.0, frozen=True)
    learnable_slope: bool = Field(False, frozen=True)
    epsilon: float = Field(1.0e-6, frozen=True)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")

    @classmethod
    def get_nn_type(cls) -> str:
        return "MeanConstrainedSigmoid"

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) == 0:
            raise ValueError("zero input is not allowed in Reducer.")
        dims_not_one = [d for d in input_dims if d != 1]
        for i in range(1, len(dims_not_one)):
            if dims_not_one[0] != dims_not_one[i]:
                raise ValueError(
                    "input dimensions should be same in Reducer. "
                    f"(given: {input_dims})"
                )
        sum_dim = sum(v for v in input_dims)
        return sum_dim

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        n_dim = self.gather_input_dims(*input_dims)
        return [n_dim, self.output_feature_dim]

    def confirm(self, self_module: IModuleSetting) -> None: ...

    @pydantic.field_validator("nodes")
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
                "nodes in MeanConstrainedSigmoid is inconsistent. "
                f"value {v} is not allowed except the head node."
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

        if nodes[-1] != self.output_feature_dim:
            raise ValueError(
                "the value of output_feature_dim and "
                "the last item of nodes are not consistent."
            )
        if nodes[0] <= 0:
            raise ValueError("Resolved nodes must be positive.")
        self.nodes = nodes
