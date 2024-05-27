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


class ConcatenatorSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    nodes: list[int] | None = Field(
        None
    )  # This property only overwritten when resolving.
    activation: str = Field("identity", frozen=True)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid")

    @classmethod
    def gather_input_dims(cls, *input_dims: int) -> int:
        assert len(input_dims) > 0
        sum_dim = sum(v for v in input_dims)
        return sum_dim

    @pydantic.field_validator("nodes")
    @classmethod
    def check_n_nodes(cls, vals: list[int]) -> list[int]:
        if vals is None:
            return vals

        if len(vals) != 2:
            raise ValueError(
                "size of nodes must be 2 in ConcatenatorSettings."
                f" input: {vals}"
            )
        return vals

    def get_nodes(self) -> list[int] | None:
        return self.nodes

    def overwrite_nodes(self, nodes: list[int]) -> None:
        self.nodes = nodes


class Concatenator(IPhlowerCoreModule, torch.nn.Module):
    @classmethod
    def from_setting(cls, setting: ConcatenatorSetting) -> Self:
        return Concatenator(**setting.__dict__)

    @classmethod
    def get_nn_name(cls):
        return "Concatenator"

    def __init__(self, activation: str, nodes: list[int] = None):
        self._nodes = nodes
        self._activation_name = activation
        self._activation_func = _utils.ActivationSelector.select(activation)

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor] = None,
    ) -> PhlowerTensor:
        return self._activation_func(torch.cat(data.values(), dim=-1))
