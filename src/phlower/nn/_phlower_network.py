import abc

import torch.nn
import torch.nn.functional as F

import dagstream

from phlower.data._bunched_data import BunchedTensorData

from phlower.base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections


class PhlowerNetworks(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._layers: list[IPhlowerLayer] = []

    def _construct(self):
        # topological-sort
        stream = dagstream.DagStream()
        _emplaces = stream.emplace(*self._layers)

        # add relationship

        # construct
        stream.construct()
        return stream
    
    def draw(self):
        ...
    
    def forward(self):
        ...



