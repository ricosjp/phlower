from __future__ import annotations

from enum import Enum
from typing import Literal

import pydantic
from pydantic import Field
from typing_extensions import Self

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)


class IsoGCNPropagationType(str, Enum):
    convolution = "convolution"  # gradient
    contraction = "contraction"  # divergent
    tensor_product = "tensor_product"
    rotation = "rotation"


class _SelfNetworkSetting(pydantic.BaseModel):
    use_network: bool = True
    activations: list[str] = Field(default_factory=lambda: [], frozen=True)
    dropouts: list[float] = Field(default_factory=lambda: [], frozen=True)
    bias: bool = Field(False, frozen=True)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")


class _CoeffNetworkSetting(pydantic.BaseModel):
    use_network: bool = True
    activations: list[str] = Field(default_factory=lambda: [], frozen=True)
    dropouts: list[float] = Field(default_factory=lambda: [], frozen=True)
    bias: bool = Field(True, frozen=True)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")


class _NeumannSetting(pydantic.BaseModel):
    use_neumann: bool = True
    factor: float = 1.0
    inversed_moment_name: str | None = None
    neumann_input_name: str | None = None

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    @pydantic.model_validator(mode="after")
    def check_inversed_moment_name_is_not_none(self) -> Self:
        if not self.use_neumann:
            return self

        if self.inversed_moment_name is None:
            raise ValueError(
                "inversed_moment_name must be determined "
                "when using neumann IsoGCN."
            )

        return self

    @pydantic.model_validator(mode="after")
    def check_valid_neumann_input_name(self) -> Self:
        if not self.use_neumann:
            return self

        if self.neumann_input_name is None:
            raise ValueError("Set neumann input name.")

        return self


class IsoGCNSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    # This property only overwritten when resolving.
    nodes: list[int] | None = Field(None)
    isoam_names: list[str] = Field(default_factory=lambda: [], frozen=True)
    propagations: list[str] = Field(default_factory=lambda: [], frozen=True)
    mul_order: Literal["ah_w", "a_hw"] = "ah_w"
    to_symmetric: bool = False
    self_network: _SelfNetworkSetting = Field(
        default_factory=lambda: _SelfNetworkSetting(use_network=False),
        frozen=True,
    )
    coefficient_network: _CoeffNetworkSetting = Field(
        default_factory=lambda: _CoeffNetworkSetting(use_network=False),
        frozen=True,
    )
    # neumann_options
    neumann_setting: _NeumannSetting = Field(
        default_factory=lambda: _NeumannSetting(use_neumann=False), frozen=True
    )

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    def get_nn_type(cls) -> str:
        return "IsoGCN"

    def confirm(self, self_module: IModuleSetting) -> None:
        input_keys = self_module.get_input_keys()
        if not self.neumann_setting.use_neumann:
            if len(input_keys) != 1:
                raise ValueError(
                    "Only one input is allowed "
                    "when neumann boundary is not considered. "
                    f"{input_keys=}"
                )
            return

        if len(input_keys) != 2:
            raise ValueError(
                "Two inputs are necessary "
                "when neumann boundary is considered. "
                f"{input_keys=}"
            )

        if self.neumann_setting.neumann_input_name not in input_keys:
            raise ValueError(
                f"neumann_input_name is not found in input_keys."
                f"neumann_input_name={self.neumann_setting.neumann_input_name}"
                f"{input_keys=}"
            )

        return

    def gather_input_dims(self, *input_dims: int) -> int:
        if (len(input_dims) == 0) or (len(input_dims) >= 3):
            raise ValueError("one or two inputs are allowed in IsoGCN.")
        return input_dims[0]

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        n_dim = self.gather_input_dims(*input_dims)
        return [n_dim, n_dim]

    @pydantic.field_validator("nodes")
    @classmethod
    def check_n_nodes(cls, vals: list[int] | None) -> list[int]:
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

    @pydantic.field_validator("isoam_names")
    @classmethod
    def check_isoam_names(cls, vals: list[int]) -> list[int]:
        if len(vals) == 0:
            raise ValueError("isoam_names is empty.")

        if len(vals) > 3:
            raise ValueError("Too many isoam_names is set.")

        return vals

    @pydantic.model_validator(mode="after")
    def check_valid_self_network(self) -> Self:
        if not self.self_network.use_network:
            return self

        if len(self.self_network.activations) > 1:
            raise ValueError(
                "Size of nodes and activations is not compatible "
                "in IsoGCNSettings."
                "In self_network, len(activations) <= 1"
                f" {self.self_network.activations=}"
            )

        if len(self.self_network.dropouts) > 1:
            raise ValueError(
                "Size of nodes and dropouts is not compatible "
                "in IsoGCNSettings."
                "In self_network, len(activations) <= 1"
                f" {self.self_network.dropouts=}"
            )

        return self

    @pydantic.model_validator(mode="after")
    def check_valid_coefficient_network(self) -> Self:
        if not self.coefficient_network.use_network:
            return self

        if len(self.coefficient_network.activations) != 0:
            if len(self.coefficient_network.activations) != len(self.nodes) - 1:
                raise ValueError(
                    "Size of nodes and activations is not compatible "
                    "in IsoGCNSettings."
                    "In coefficient_network, len(activations) == len(nodes) - 1"
                    f" {self.coefficient_network.activations=}"
                )

        if len(self.coefficient_network.dropouts) != 0:
            if len(self.coefficient_network.dropouts) != len(self.nodes) - 1:
                raise ValueError(
                    "Size of nodes and dropouts is not compatible "
                    "in IsoGCNSettings."
                    "In coefficient_network, len(activations) == len(nodes) - 1"
                    f" {self.coefficient_network.dropouts=}"
                )

        return self

    @pydantic.model_validator(mode="after")
    def check_self_weight_when_use_neumann(self) -> Self:
        if not self.neumann_setting.use_neumann:
            return self

        if not self.self_network.use_network:
            raise ValueError(
                "Use self_network when neumannn layer is necessary. "
                "It is because neumann layer refers to weight of self_network."
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
