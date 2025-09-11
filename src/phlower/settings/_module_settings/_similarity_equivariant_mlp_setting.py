from __future__ import annotations

import pydantic
from pydantic import Field
from typing_extensions import Self

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)
from phlower.utils.enums import PhysicalDimensionSymbolType


class SimilarityEquivariantMLPSetting(
    IPhlowerLayerParameters, pydantic.BaseModel
):
    nodes: list[int] = Field(
        ...
    )  # This property only overwritten when resolving.
    scale_names: dict[str, str] = Field(default_factory=lambda: {}, frozen=True)
    activations: list[str] = Field(default_factory=lambda: [], frozen=True)
    dropouts: list[float] = Field(default_factory=lambda: [], frozen=True)
    bias: bool = Field(False, frozen=True)
    create_linear_weight: bool = Field(False, frozen=True)
    norm_function_name: str = Field(
        default_factory=lambda: "identity", frozen=True
    )
    disable_en_equivariance: bool = Field(False, frozen=True)
    invariant: bool = Field(False, frozen=True)
    centering: bool = Field(False, frozen=True)
    cross_interaction: bool = Field(False, frozen=True)
    normalize: bool = Field(True, frozen=True)

    model_config = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    def get_nn_type(cls) -> str:
        return "SimilarityEquivariantMLP"

    def confirm(self, self_module: IModuleSetting) -> None:
        return

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) != 1:
            raise ValueError(
                "Only one input is allowed in SimilarityEquivariantMLP."
            )
        return input_dims[0]

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        n_dim = self.gather_input_dims(*input_dims)
        return [n_dim, n_dim]

    @pydantic.field_validator("nodes")
    @classmethod
    def check_n_nodes(cls, vals: list[int]) -> list[int]:
        if len(vals) < 2:
            raise ValueError(
                "size of nodes must be larger than 1 in "
                f"SimilarityEquivariantMLPSetting. input: {vals}"
            )

        for i, v in enumerate(vals):
            if v > 0:
                continue

            if (i == 0) and (v == -1):
                continue

            raise ValueError(
                "nodes in SimilarityEquivariantMLPSetting is inconsistent. "
                f"value {v} in {i}-th of nodes is not allowed."
            )

        return vals

    @pydantic.model_validator(mode="after")
    def check_nodes_size(self) -> Self:
        if len(self.nodes) - 1 != len(self.activations):
            raise ValueError(
                "Size of nodes and activations is not compatible "
                "in SimilarityEquivariantMLPSetting."
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

    @pydantic.field_validator("scale_names")
    @classmethod
    def check_scale_names(cls, vals: dict[str, str]) -> list[int]:
        if len(vals) == 0:
            raise ValueError("scale_names is empty.")

        for key in vals.keys():
            if not PhysicalDimensionSymbolType.is_exist(key):
                raise ValueError(
                    f"Key not in physical dimension: {vals.keys()}"
                )

        return vals
