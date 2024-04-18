import dagstream
import torch.nn

from phlower.nn._interface_layer import IPhlowerLayer

# TODO: UNDER CONSTRUCTION


class PhlowerNetworks(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._layers: list[IPhlowerLayer] = []

    def _construct(self):
        # topological-sort
        stream = dagstream.DagStream()
        # _emplaces = stream.emplace(*self._layers)

        # add relationship

        # construct
        stream.construct()
        return stream

    def draw(self): ...

    def forward(self): ...
