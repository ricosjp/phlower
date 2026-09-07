from __future__ import annotations

import torch
from phlower_tensor import ISimulationField, PhlowerTensor
from phlower_tensor.collections import IPhlowerTensorCollections

from phlower.nn._functionals._edge import extract_edge_indices
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import EdgeDifferenceSetting


class EdgeDifference(IPhlowerCoreModule, torch.nn.Module):
    """EdgeDifference computes the difference between the nodal features
    of the two endpoints of each edge:
    ``e_k = h_j - h_i`` for the k-th edge (i, j).
    Self-loop edges are ignored.

    input: ([t,] n_nodes, ..., f)
    output: ([t,] n_edges, ..., f)
    ``t`` is optional.

    When ``with_norm`` is True, the L2 norm of the difference is
    concatenated to the output, which then has shape
    ([t,] n_edges, f + 1).

    Applied to node positions, this yields the relative position
    and distance edge features used in MeshGraphNets.

    Parameters
    ----------
    support_name: str
        Name of the support tensor defining the edges.
    with_norm: bool
        Whether to concatenate the L2 norm of the difference to the output.
        Only available for rank-0 tensors.
        Defaults to False.

    Examples
    --------
    >>> edge_difference = EdgeDifference(
    ...     support_name="support", with_norm=True
    ... )
    >>> edge_difference(data, field_data=field_data)
    """

    @classmethod
    def from_setting(cls, setting: EdgeDifferenceSetting) -> EdgeDifference:
        """Create EdgeDifference from setting object

        Args:
            setting (EdgeDifferenceSetting): setting object

        Returns:
            EdgeDifference: EdgeDifference object
        """
        return EdgeDifference(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "EdgeDifference"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        support_name: str,
        with_norm: bool = False,
        nodes: list[int] | None = None,
    ) -> None:
        super().__init__()

        self._support_name = support_name
        self._with_norm = with_norm
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
            raise ValueError("EdgeDifference does not support voxel tensors.")
        if self._with_norm and h.rank() != 0:
            raise ValueError(
                "with_norm in EdgeDifference is only applicable to "
                f"a rank-0 tensor. actual rank: {h.rank()}"
            )

        receiver, sender = extract_edge_indices(support)  # each (n_edges,)
        # ([t,] n_nodes, ...) -> ([t,] n_edges, ...)
        if h.is_time_series:
            diff = h[:, sender] - h[:, receiver]
        else:
            diff = h[sender] - h[receiver]
        if self._with_norm:
            # (..., f) -> (..., f + 1)
            norm = torch.linalg.vector_norm(diff, dim=-1, keepdim=True)
            diff = torch.cat([diff, norm], dim=-1)
        return diff
