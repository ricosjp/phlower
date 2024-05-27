from __future__ import annotations

import pydantic
import torch
from pydantic import Field
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._interface_layer import (
    IPhlowerCoreModule,
    IPhlowerLayerParameters,
)
from phlower.nn._modules import _utils


class MLPSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    nodes: list[int] = Field(
        ...
    )  # This property only overwritten when resolving.
    activations: list[str] = Field(default_factory=lambda: [], frozen=True)
    dropouts: list[float] = Field(default_factory=lambda: [], frozen=True)
    bias: bool = Field(False, frozen=True)

    @classmethod
    def gather_input_dims(cls, *input_dims: int) -> int:
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

    def get_nodes(self) -> list[int] | None:
        return self.nodes

    def overwrite_nodes(self, nodes: list[int]) -> None:
        self.nodes = nodes


class MLP(IPhlowerCoreModule, torch.nn.Module):
    """Multi Layer Perceptron."""

    @classmethod
    def from_setting(cls, setting: MLPSetting) -> Self:
        return MLP(**setting.__dict__)

    @classmethod
    def get_nn_name(cls):
        return "MLP"

    def __init__(
        self,
        nodes: list[int],
        activations: list[str] = None,
        dropouts: list[float] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if activations is None:
            activations = []
        if dropouts is None:
            dropouts = []

        self._chains = _utils.ExtendedLinearList(
            nodes=nodes, activations=activations, dropouts=dropouts, bias=bias
        )
        self._nodes = nodes
        self._activations = activations

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor] = None,
    ) -> PhlowerTensor:
        h = data.unique_item()
        for i in range(len(self._chains)):
            h = self._chains.forward(h, index=i)
        return h
