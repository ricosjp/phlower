from __future__ import annotations

import pydantic
import pydantic.dataclasses as dc
import torch
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._interface_layer import (
    IPhlowerCoreModule,
    IPhlowerLayerParameters,
)
from phlower.nn._modules import _utils


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class GCNResolvedSetting(IPhlowerLayerParameters):
    nodes: list[int]
    support_name: str
    repeat: int = 1
    factor: float = 1.0
    activations: list[str] = pydantic.Field(default_factory=lambda: [])
    dropouts: list[float] = pydantic.Field(default_factory=lambda: [])
    bias: bool = False

    @pydantic.field_validator("nodes")
    @classmethod
    def check_n_nodes(cls, v: list[int]) -> list[int]:
        if len(v) < 2:
            raise ValueError(
                "size of nodes must be larger than 1 in GCNSettings."
                f" input: {len(v)}"
            )
        return v

    @pydantic.model_validator(mode="after")
    def check_nodes_size(self) -> Self:
        if len(self.nodes) - 1 != len(self.activations):
            raise ValueError(
                "Size of nodes and activations is not compatible "
                "in GCNSettings."
                " len(nodes) must be equal to 1 + len(activations)."
            )
        return self


class GCN(IPhlowerCoreModule, torch.nn.Module):
    @classmethod
    def create(cls, setting: GCNResolvedSetting) -> GCN:
        return GCN(**setting.__dict__)

    def __init__(
        self,
        nodes: list[int],
        support_name: str,
        activations: list[str] = None,
        dropouts: list[float] = None,
        repeat: int = 1,
        factor: float = 1.0,
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
        self._support_name = support_name
        self._repeat = repeat
        self._factor = factor

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor],
    ) -> PhlowerTensor:
        support = supports[self._support_name]
        h = data.unique_item()
        for i in range(len(self._chains)):
            h = self._propagate(h, support)
            h = self._chains.forward(h, index=i)
        return h

    def _propagate(
        self, x: PhlowerTensor, support: PhlowerTensor
    ) -> PhlowerTensor:
        n_node = x.shape[0]
        h = torch.reshape(x, (n_node, -1))
        for _ in range(self._repeat):
            h = torch.sparse.mm(support, h) * self._factor
        return torch.reshape(h, x.shape)
