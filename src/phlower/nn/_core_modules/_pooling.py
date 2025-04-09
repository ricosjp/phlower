from __future__ import annotations

from logging import getLogger

import torch

from phlower._base import phlower_tensor
from phlower._base._functionals import unbatch
from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._functionals import PoolingSelector
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import PoolingSetting

_logger = getLogger(__name__)


class Pooling(IPhlowerCoreModule, torch.nn.Module):
    @classmethod
    def from_setting(cls, setting: PoolingSetting) -> Pooling:
        """Create Pooling from setting object

        Args:
            setting (PoolingSetting): setting object

        Returns:
            Pooling: Pooling Module
        """
        return Pooling(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of Pooling

        Returns:
            str: name
        """
        return "Pooling"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        pool_operator_name: str,
        nodes: list[int] | None = None,
        unbatch_key: str | None = None,
    ):
        super().__init__()
        self._nodes = nodes
        self._pool_operator_name = pool_operator_name
        self._unbatch_key = unbatch_key

        self._pooling_operator = PoolingSelector.select(pool_operator_name)

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None: ...

    def get_reference_name(self) -> str | None:
        return None

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None = None,
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

        inputs = data.unique_item()
        node_dim = inputs.shape_pattern.nodes_dim
        unbatched_targets = self._unbatch(inputs, field_data=field_data)
        results = torch.cat(
            [self._pooling(target) for target in unbatched_targets],
            dim=node_dim,
        )

        return results

    def _pooling(self, target: PhlowerTensor) -> PhlowerTensor:
        _value = self._pooling_operator(
            [target.to_tensor()], target.shape_pattern.nodes_dim
        )
        return phlower_tensor(
            tensor=_value,
            dimension=target.dimension,
            is_time_series=target.is_time_series,
            is_voxel=target.is_voxel,
        )

    def _unbatch(
        self, target: PhlowerTensor, field_data: ISimulationField | None
    ) -> list[PhlowerTensor]:
        if (field_data is None) or (self._unbatch_key is None):
            _logger.info(
                "batch info is not passed to DeepSets. "
                "Unbatch operation is skipped."
            )
            return [target]

        return unbatch(
            target,
            n_nodes=field_data.get_batched_n_nodes(self._unbatch_key),
        )
