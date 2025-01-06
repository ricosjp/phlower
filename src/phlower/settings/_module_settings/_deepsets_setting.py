from __future__ import annotations

from typing import Literal

import pydantic
from pydantic import Field
from typing_extensions import Self

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)
from phlower.utils.enums import ActivationType


class DeepSetsSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    # This property only overwritten when resolving.
    nodes: list[int]
    unbatch_key: str = Field(frozen=True)
    activations: list[str] = Field(default_factory=lambda: [], frozen=True)
    dropouts: list[float] = Field(default_factory=lambda: [], frozen=True)
    bias: bool = Field(True, frozen=True)

    last_activation: str = Field("identity", frozen=True)
    pool_operator: Literal["max", "mean"] = Field("max", frozen=True)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")

    def confirm(self, self_module: IModuleSetting) -> None:
        return

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) != 1:
            raise ValueError("only one input is allowed in DeepSets.")
        return input_dims[0]

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        n_dim = self.gather_input_dims(*input_dims)
        return [n_dim, n_dim]

    @pydantic.field_validator("nodes", mode="before")
    @classmethod
    def check_n_nodes(cls, vals: list[int]) -> list[int]:
        if len(vals) < 2:
            raise ValueError(
                "size of nodes must be larger than 1 in DeepSetsSettings."
                f" input: {vals}"
            )

        for i, v in enumerate(vals):
            if v > 0:
                continue

            if (i == 0) and (v == -1):
                continue

            raise ValueError(
                "nodes in DeepSets is inconsistent. "
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
    def check_existing_activation_name(self) -> Self:
        if self.last_activation not in ActivationType.__members__:
            raise ValueError(f"{self.last_activation} is not implemented.")
        return self

    @pydantic.model_validator(mode="after")
    def check_nodes_size(self) -> Self:
        if len(self.nodes) - 1 != len(self.activations):
            raise ValueError(
                "Size of nodes and activations is not compatible "
                "in DeepSetsSettings."
                " len(nodes) must be equal to 1 + len(activations)."
            )

        if len(self.nodes) - 1 != len(self.dropouts):
            raise ValueError(
                "Size of nodes and dropouts is not compatible "
                "in DeepSetsSettings."
                " len(nodes) must be equal to 1 + len(dropouts)."
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
        self.nodes = nodes
