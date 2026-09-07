from __future__ import annotations

import torch
from phlower_tensor import ISimulationField, PhlowerTensor
from phlower_tensor.collections import IPhlowerTensorCollections

from phlower.nn._functionals._edge import extract_edge_indices
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import EdgeToNodeSumSetting


class EdgeToNodeSum(IPhlowerCoreModule, torch.nn.Module):
    """EdgeToNodeSum sums the edge feature onto the receiver node
    of each edge:
    ``v_i = sum_k e_k`` over the edges (i, j) with receiver i.
    Self-loop edges are ignored.

    input: ([t,] n_edges, ..., f)
    output: ([t,] n_nodes, ..., f)
    ``t`` is optional.

    The input must be edge features whose ``n_edges`` axis follows
    the edge ordering of the support, such as the output of
    EdgeDifference or EdgeGather with the same support.

    Parameters
    ----------
    support_name: str
        Name of the support tensor defining the edges.

    Examples
    --------
    >>> edge_to_node_sum = EdgeToNodeSum(support_name="support")
    >>> edge_to_node_sum(data, field_data=field_data)
    """

    @classmethod
    def from_setting(cls, setting: EdgeToNodeSumSetting) -> EdgeToNodeSum:
        """Create EdgeToNodeSum from setting object

        Args:
            setting (EdgeToNodeSumSetting): setting object

        Returns:
            EdgeToNodeSum: EdgeToNodeSum object
        """
        return EdgeToNodeSum(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "EdgeToNodeSum"

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
        edge = data.unique_item()
        if edge.is_voxel:
            raise ValueError("EdgeToNodeSum does not support voxel tensors.")

        receiver, _ = extract_edge_indices(support)  # (n_edges,)
        node_dim = edge.shape_pattern.nodes_dim
        if edge.shape[node_dim] != receiver.shape[0]:
            raise ValueError(
                "The number of edges in the input does not match "
                f"that of the support. input: {edge.shape[node_dim]}, "
                f"support: {receiver.shape[0]}"
            )

        edge_tensor = edge.to_tensor()
        # edge: ([t,] n_edges, ...) -> summed: ([t,] n_nodes, ...)
        shape = list(edge_tensor.shape)
        shape[node_dim] = support.shape[0]
        summed = edge_tensor.new_zeros(shape).index_add(
            node_dim, receiver, edge_tensor
        )
        return PhlowerTensor.from_pattern(
            summed,
            dimension_tensor=edge.dimension,
            pattern=edge.shape_pattern,
        )
