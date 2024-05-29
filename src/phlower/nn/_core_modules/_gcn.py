from __future__ import annotations

import pydantic
import torch
from pydantic import Field
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._interface_module import (
    IPhlowerCoreModule
)
from phlower.nn._core_modules import _utils
from phlower.settings._module_settings import GCNSetting


class GCN(IPhlowerCoreModule, torch.nn.Module):
    @classmethod
    def from_setting(cls, setting: GCNSetting) -> GCN:
        return GCN(**setting.__dict__)

    @classmethod
    def get_nn_name(cls):
        return "GCN"

    @classmethod
    def get_parameter_setting_cls(cls):
        return GCNSetting

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
