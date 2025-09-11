from __future__ import annotations

import pydantic
from pydantic import Field
from typing_extensions import Self

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)
from phlower.utils.enums import ActivationType


class EnEquivariantMLPSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    nodes: list[int]
    activations: list[str] = Field(default_factory=lambda: [], frozen=True)
    dropouts: list[float] = Field(default_factory=lambda: [], frozen=True)
    bias: bool = Field(True, frozen=True)
    create_linear_weight: bool = Field(False, frozen=True)
    norm_function_name: str = Field(
        default_factory=lambda: "identity", frozen=True
    )

    # If cross_interaction = True, use interaction layer proposed by
    # https://arxiv.org/abs/2203.06442. If normalize = True, use eq (12)
    # in that paper.
    cross_interaction: bool = Field(False, frozen=True)
    normalize: bool = Field(True, frozen=True)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    def get_nn_type(cls) -> str:
        return "EnEquivariantMLP"

    def confirm(self, self_module: IModuleSetting) -> None: ...

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) != 1:
            raise ValueError("Only one input is allowed in EnEquivariantMLP.")
        return input_dims[0]

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        n_dim = self.gather_input_dims(*input_dims)
        return [n_dim, self.nodes[-1]]

    @pydantic.field_validator("nodes")
    @classmethod
    def check_n_nodes(cls, vals: list[int]) -> list[int]:
        if len(vals) < 2:
            raise ValueError(
                "size of nodes must be larger than 1 in "
                f"EnEquivariantMLPSetting. input: {vals}"
            )

        for i, v in enumerate(vals):
            if v > 0:
                continue

            if (i == 0) and (v == -1):
                continue

            raise ValueError(
                "nodes in EnEquivariantMLPSetting is inconsistent. "
                f"value {v} in {i}-th of nodes is not allowed."
            )

        return vals

    @pydantic.model_validator(mode="before")
    @classmethod
    def fill_empty_activations_dropouts(cls, values: dict) -> dict:
        n_nodes = len(values.get("nodes"))
        activations = values.get("activations", [])
        dropouts = values.get("dropouts", [])

        if len(activations) == 0:
            values["activations"] = ["identity" for _ in range(n_nodes - 1)]

        if len(dropouts) == 0:
            values["dropouts"] = [0 for _ in range(n_nodes - 1)]

        return values

    @pydantic.model_validator(mode="after")
    def check_nodes_size(self) -> Self:
        if len(self.nodes) - 1 != len(self.activations):
            raise ValueError(
                "Size of nodes and activations is not compatible "
                "in EnEquivariantMLPSetting."
                " len(nodes) must be equal to 1 + len(activations)."
            )
        return self

    @pydantic.model_validator(mode="after")
    def check_exist_norm_function_name(self) -> Self:
        if self.norm_function_name not in ActivationType.__members__:
            raise ValueError(
                f"{self.norm_function_name} is not implemented in "
                "EnEquivariantMLPSetting."
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
