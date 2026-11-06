from __future__ import annotations

import torch
from phlower_tensor import ISimulationField, PhlowerTensor, phlower_tensor
from phlower_tensor.collections import IPhlowerTensorCollections
from phlower_tensor.functionals import unbatch

from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import NodalExpandSetting


class NodalExpand(IPhlowerCoreModule, torch.nn.Module):
    """Expand an input to the node count of the named reference input.

    Parameters
    ----------
    reference_key : str
        The key of the reference input to match the node count.
    nodes : list[int] | None, optional
        The node configuration for the expanded output, by default None.
    unbatch_key : str | None, optional
        The key used for unbatching the input tensors, by default None.

    Notes
    -----
    The source input must have exactly one node.
    Voxel tensors are not supported.

    Examples
    --------
    >>> source = phlower_tensor(torch.randn(1, 3))
    >>> reference = phlower_tensor(torch.randn(5, 3))
    >>> expanded = _expand(source, reference)
    >>> expanded.shape
    torch.Size([5, 3])
    """

    @classmethod
    def from_setting(cls, setting: NodalExpandSetting) -> NodalExpand:
        return cls(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        return "NodalExpand"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        reference_key: str,
        nodes: list[int] | None = None,
        unbatch_key: str | None = None,
    ) -> None:
        super().__init__()
        self._reference_key = reference_key
        self._nodes = nodes
        self._unbatch_key = unbatch_key

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwargs
    ) -> None: ...

    def get_reference_name(self) -> str | None:
        return None

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None = None,
        **kwargs,
    ) -> PhlowerTensor:
        if len(data) != 2:
            raise ValueError("Exactly two inputs are required in NodalExpand.")

        if self._reference_key not in data:
            raise ValueError(
                f"reference_key {self._reference_key!r} is not found in inputs."
            )
        reference = data.pop(self._reference_key)
        source = data.unique_item()

        if source.is_voxel or reference.is_voxel:
            raise ValueError("NodalExpand does not support voxel tensors.")

        if self._unbatch_key is None:
            return _expand(source, reference)

        if field_data is None:
            raise ValueError(
                "Unbatch key is specified, but field data is not provided."
            )
        n_nodes = field_data.get_batched_n_nodes(self._unbatch_key)
        sources = unbatch(source, n_nodes=n_nodes)
        references = unbatch(reference, n_nodes=n_nodes)
        return torch.cat(
            [
                _expand(src, ref)
                for src, ref in zip(sources, references, strict=True)
            ],
            dim=source.shape_pattern.nodes_dim,
        )


def _expand(source: PhlowerTensor, reference: PhlowerTensor) -> PhlowerTensor:
    if source.shape_pattern.get_n_vertices() != 1:
        raise ValueError(
            "NodalExpand requires the source to have exactly one node."
        )

    node_dim = source.shape_pattern.nodes_dim
    n_nodes = reference.shape[reference.shape_pattern.nodes_dim]
    shape = list(source.shape)
    shape[node_dim] = n_nodes

    return phlower_tensor(
        source.to_tensor().expand(*shape),
        dimension=source.dimension,
        is_time_series=source.is_time_series,
        is_voxel=source.is_voxel,
    )
