from __future__ import annotations

from enum import Enum
from typing import Literal

import pydantic
from pydantic import Field
from typing_extensions import Self

from phlower.settings._interface import (
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)


class IsoGCNPropagationType(Enum, str):
    convolution = "convolution"  # gradient
    contraction = "conttraction"  # divergent
    rotation = "rotation"


class _SubNetworkSetting(pydantic.BaseModel):
    is_active: bool = True
    activations: list[str] = Field(default_factory=lambda: [], frozen=True)
    dropouts: list[float] = Field(default_factory=lambda: [], frozen=True)
    bias: bool = Field(False, frozen=True)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")


class IsoGCNNeumannLinearType(Enum, str):
    identity = "identity"
    reuse_graph_weight = "reuse_graph_weight"
    create_new = "create_new"


class _NeumannSetting(pydantic.BaseModel):
    is_active: bool = True
    linear_model_type: IsoGCNNeumannLinearType = (
        IsoGCNNeumannLinearType.identity
    )
    factor: float = 1.0
    apply_sigmoid_ratio: bool = False
    inversed_moment_name: str | None = None

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    @pydantic.model_validator(mode="after")
    def check_inversed_moment_name_is_not_none(self) -> Self:
        if not self.is_active:
            return self

        assert (
            self.inversed_moment_name is not None
        ), "inversed_moment_name must be determined when using neumann IsoGCN."


class IsoGCNSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    # This property only overwritten when resolving.
    nodes: list[int] = Field(...)
    support_names: list[str] = Field(..., frozen=True)
    propagations: list[str] = Field(default_factory=lambda: [], frozen=True)
    mul_order: Literal["ah_w", "a_hw"] = "ah_w"
    graph_weight: _SubNetworkSetting = Field(
        default_factory=lambda: _SubNetworkSetting(), frozen=True
    )
    coefficient_network: _SubNetworkSetting = Field(
        default_factory=lambda: _SubNetworkSetting(), frozen=True
    )
    # neumann_options
    neumann_setting: _NeumannSetting = Field(
        default_factory=lambda: _NeumannSetting(is_active=False), frozen=True
    )

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")

    def gather_input_dims(self, *input_dims: int) -> int:
        if len(input_dims) != 1:
            raise ValueError("only one input is allowed in GCN.")
        return input_dims[0]

    @pydantic.field_validator("nodes")
    @classmethod
    def check_n_nodes(cls, vals: list[int]) -> list[int]:
        if len(vals) < 2:
            raise ValueError(
                "size of nodes must be larger than 1 in IsoGCNSettings."
                f" input: {vals}"
            )

        for i, v in enumerate(vals):
            if v > 0:
                continue

            if (i == 0) and (v == -1):
                continue

            raise ValueError(
                "nodes in IsoGCN is inconsistent. "
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
                "in GCNSettings."
                " len(nodes) must be equal to 1 + len(activations)."
            )

        if len(self.nodes) - 1 != len(self.dropouts):
            raise ValueError(
                "Size of nodes and dropouts is not compatible "
                "in GCNSettings."
                " len(nodes) must be equal to 1 + len(dropouts)."
            )
        return self

    def get_n_nodes(self) -> list[int]:
        return self.nodes

    def overwrite_nodes(self, nodes: list[int]) -> None:
        self.nodes = nodes

    @property
    def need_reference(self) -> bool:
        return False

    def get_reference(self, parent: IReadOnlyReferenceGroupSetting) -> None:
        return
