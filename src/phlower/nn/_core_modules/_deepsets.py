from __future__ import annotations

from collections.abc import Callable

import torch

from phlower._base._functionals import unbatch
from phlower._base.tensors import (
    PhlowerTensor,
    phlower_tensor,
)
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _utils
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import DeepSetsSetting
from phlower.utils import get_logger

_logger = get_logger(__name__)


def _max_pool(tensors: list[torch.Tensor], nodes_dim: int) -> torch.Tensor:
    return torch.stack(
        [torch.max(t, dim=nodes_dim, keepdim=False)[0] for t in tensors],
        dim=nodes_dim,
    )


def _mean(tensors: list[torch.Tensor], nodes_dim: int) -> torch.Tensor:
    return torch.stack(
        [torch.mean(t, dim=nodes_dim, keepdim=False) for t in tensors],
        dim=nodes_dim,
    )


_PoolOperator = Callable[[list[torch.Tensor], int], torch.Tensor]


class DeepSets(IPhlowerCoreModule, torch.nn.Module):
    """
    DeepSets

    Ref: https://arxiv.org/abs/1703.06114
    """

    _REGISTERED_POOL_OP: dict[str, _PoolOperator] = {
        "max": _max_pool,
        "mean": _mean,
    }

    @classmethod
    def from_setting(cls, setting: DeepSetsSetting) -> DeepSets:
        """Create DeepSets from setting object

        Args:
            setting (DeepSetsSetting): setting object

        Returns:
            Self: DeepSets
        """
        config = _utils.MLPConfiguration(
            nodes=setting.nodes,
            activations=setting.activations,
            dropouts=setting.dropouts,
            bias=setting.bias,
        )

        # So far, lambda and gamma are assumed to be the same config
        return DeepSets(
            lambda_config=config,
            gamma_config=config,
            last_activation_name=setting.last_activation,
            pool_operator_name=setting.pool_operator,
            unbatch_key=setting.unbatch_key,
        )

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of DeepSets

        Returns:
            str: name
        """
        return "DeepSets"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        lambda_config: _utils.MLPConfiguration,
        gamma_config: _utils.MLPConfiguration,
        last_activation_name: str,
        pool_operator_name: str,
        unbatch_key: str | None = None,
    ):
        super().__init__()

        self._lambda = _utils.ExtendedLinearList.from_config(lambda_config)
        self._gamma = _utils.ExtendedLinearList.from_config(gamma_config)
        self._last_activation_name = last_activation_name
        self._pool_operator_name = pool_operator_name

        self._last_activation = _utils.ActivationSelector.select(
            last_activation_name
        )
        self._pool_operator = self._select_pool_operator(pool_operator_name)
        self._reference_batch_name = unbatch_key

    def _select_pool_operator(self, name: str) -> _PoolOperator:
        if name not in self._REGISTERED_POOL_OP:
            raise NotImplementedError(
                f"pooling operation named {name} is not defined."
            )

        return self._REGISTERED_POOL_OP[name]

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
            data (IPhlowerTensorCollections):
                data which receives from predecessors
            field_data (ISimulationField):
                Constant information through training or prediction

        Returns:
            PhlowerTensor: Tensor object
        """

        inputs = data.unique_item()
        node_dim = inputs.shape_pattern.nodes_dim
        unbatched_targets = self._unbatch(inputs, field_data=field_data)

        results = torch.cat(
            [
                self._lambda.forward(target)
                + self._pooling(self._gamma.forward(target))
                for target in unbatched_targets
            ],
            dim=node_dim,
        )

        return self._last_activation(results)

    def _pooling(self, target: PhlowerTensor) -> PhlowerTensor:
        _value = self._pool_operator(
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
        if (field_data is None) or (self._reference_batch_name is None):
            _logger.info(
                "batch info is not passed to DeepSets. "
                "Unbatch operation is skipped."
            )
            return [target]

        return unbatch(
            target,
            n_nodes=field_data.get_batched_n_nodes(self._reference_batch_name),
        )
