from __future__ import annotations

import torch
from phlower_tensor import ISimulationField, PhlowerTensor
from phlower_tensor.collections import IPhlowerTensorCollections

from phlower.nn._functionals._edge import extract_edge_indices
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import EdgeGatherSetting


class EdgeGather(IPhlowerCoreModule, torch.nn.Module):
    """EdgeGather concatenates the nodal features
    of the two endpoints of each edge:
    ``e_k = [h_i, h_j]`` for the k-th edge (i, j).
    Self-loop edges are ignored.

    input: ([t,] n_nodes, ..., f)
    output: ([t,] n_edges, ..., 2 * f)
    ``t`` is optional.

    Combined with EdgeDifference and EdgeToNodeSum, this allows
    message-passing blocks such as MeshGraphNets to be assembled
    from feature-wise modules like MLP.

    Parameters
    ----------
    support_name: str
        Name of the support tensor defining the edges.

    Examples
    --------
    >>> edge_gather = EdgeGather(support_name="support")
    >>> edge_gather(data, field_data=field_data)
    """

    @classmethod
    def from_setting(cls, setting: EdgeGatherSetting) -> EdgeGather:
        """Create EdgeGather from setting object

        Args:
            setting (EdgeGatherSetting): setting object

        Returns:
            EdgeGather: EdgeGather object
        """
        return EdgeGather(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "EdgeGather"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        support_name: str,
        nodes: list[int] | None = None,
    ) -> None:
        super().__init__()

        self._support_name = support_name
        self._nodes = nodes

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None: ...

    def get_reference_name(self) -> str | None:
        return None

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField,
        **kwards,
    ) -> PhlowerTensor:
        """forward function which overloads torch.nn.Module

        Args:
            data: IPhlowerTensorCollections
                data which receives from predecessors
            field_data: ISimulationField | None
                Constant information through training or prediction

        Returns:
            PhlowerTensor: Tensor object
        """
        support = field_data[self._support_name]  # (n_nodes, n_nodes)
        h = data.unique_item()
        if h.is_voxel:
            raise ValueError("EdgeGather does not support voxel tensors.")

        receiver, sender = extract_edge_indices(support)  # each (n_edges,)
        # ([t,] n_nodes, ..., f) -> ([t,] n_edges, ..., 2 * f)
        if h.is_time_series:
            return torch.cat([h[:, receiver], h[:, sender]], dim=-1)
        return torch.cat([h[receiver], h[sender]], dim=-1)
