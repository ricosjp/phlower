from __future__ import annotations

from collections.abc import ItemsView, KeysView
from typing import TYPE_CHECKING

import pyvista as pv
import torch
from phlower_tensor import (
    GraphBatchInfo,
    ISimulationField,
    PhlowerTensor,
    PhysicalDimensions,
    SimulationField,
)
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.utils._lazy_import import _LazyImport

if TYPE_CHECKING:
    import graphlow
else:
    # NOTE: From Python 3.15, we can use `lazy import` directly.
    # For now, we use a custom lazy import mechanism.
    graphlow = _LazyImport("graphlow")


class PyVistaMeshAdapter:
    def __init__(
        self, pv_mesh: pv.DataSet, dimensions: dict[str, PhysicalDimensions]
    ) -> None:
        self._pv_mesh = pv_mesh
        self._dimensions = dimensions

    def get_dimension_collection(self) -> dict[str, PhysicalDimensions]:
        return self._dimensions

    def get_pvmesh(self) -> pv.DataSet:
        return self._pv_mesh


def create_simulation_field(
    field_tensors: dict[str, PhlowerTensor],
    batch_info: dict[str, GraphBatchInfo],
    tensor_mesh: PyVistaMeshAdapter | None = None,
    disable_dimensions: bool = False,
) -> SimulationField:

    if tensor_mesh is None:
        return SimulationField(
            field_tensors=phlower_tensor_collection(field_tensors),
            batch_info=batch_info,
        )

    else:
        return GraphlowSimulationField.from_pyvista_adapter(
            field_tensors=phlower_tensor_collection(field_tensors),
            mesh=tensor_mesh,
            batch_info=batch_info,
            disable_dimensions=disable_dimensions,
        )


# NOTE: This class is an adapter for graphlow's TensorMesh to be used
# as a simulation field in phlower including batch operation.
# Currently, it only supports non-batched simulation fields.
class GraphlowSimulationField(ISimulationField):
    @classmethod
    def from_pyvista_adapter(
        cls,
        mesh: PyVistaMeshAdapter,
        field_tensors: dict[str, PhlowerTensor],
        batch_info: dict[str, GraphBatchInfo] | None = None,
        disable_dimensions: bool = False,
    ) -> GraphlowSimulationField:
        _mesh = graphlow.from_pyvista(
            mesh.get_pvmesh(),
            backend="phlower",
            disable_dimensions=disable_dimensions,
            dimension_collection=mesh.get_dimension_collection(),
        )
        return GraphlowSimulationField(
            mesh=_mesh, field_tensors=field_tensors, batch_info=batch_info
        )

    def __init__(
        self,
        field_tensors: IPhlowerTensorCollections,
        mesh: graphlow.TensorMesh,
        batch_info: dict[str, GraphBatchInfo] | None = None,
    ) -> None:
        self._mesh = mesh
        self._batch_info = batch_info or {}
        self._field_tensors = field_tensors

        if len(self._batch_info) > 1:
            raise NotImplementedError(
                "Batch operation for graphlow simulation field "
                "is not implemented yet."
                f"{self._batch_info=}"
            )

    def get_mesh(self) -> graphlow.TensorMesh:
        return self._mesh

    def keys(self) -> KeysView[str]:
        return self._mesh.point_data.keys() | self._field_tensors.keys()

    def items(self) -> ItemsView[str, PhlowerTensor]:
        return self._mesh.point_data.items() | self._field_tensors.items()

    def __getitem__(self, name: str) -> PhlowerTensor:
        if name in self._mesh.point_data:
            return self._mesh.point_data[name]

        if name in self._field_tensors:
            return self._field_tensors[name]

        raise KeyError(f"{name} is not found in simulation field.")

    def __contains__(self, name: str) -> bool:
        return name in self._mesh.point_data or name in self._field_tensors

    def get_batch_info(self, name: str) -> GraphBatchInfo:
        if name not in self._batch_info:
            raise KeyError(f"{name} is not found in simulation field.")
        return self._batch_info[name]

    def get_batched_n_nodes(self, name: str) -> tuple[int]:
        if not self._batch_info:
            raise ValueError("Information about batch is not found.")

        # NOTE: Assume that batch information is same among features.
        batch_info = self.get_batch_info(name)
        return batch_info.n_nodes

    def to(
        self, device: str | torch.device, non_blocking: bool = False
    ) -> GraphlowSimulationField:
        return GraphlowSimulationField(
            field_tensors=self._field_tensors.to(
                device=device, non_blocking=non_blocking
            ),
            mesh=self._mesh.to(device=device),
            batch_info=self._batch_info,
        )
