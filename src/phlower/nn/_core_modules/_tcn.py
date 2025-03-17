from __future__ import annotations

import torch

from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules._utils import ExtendedConv1DList
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import TCNSetting


class TCN(torch.nn.Module, IPhlowerCoreModule):
    """
    Temporal Convolutional Networks

    Ref. https://arxiv.org/abs/1803.01271

    Parameters
    ----------
    nodes: list[int]
        List of feature dimension sizes (The last value of tensor shape).
    kernel_sizes: list[int]
        List of kernel sizes.
    dilations: list[int]
        List of dilations.
    activations: list[str]
        List of activation functions.
    bias: bool
        Whether to use bias.
    dropouts: list[float] | None (optional)
        List of dropout rates.

    Examples
    --------
    >>> tcn = TCN(
    ...     nodes=[10, 20, 30],
    ...     kernel_sizes=[3, 3, 3],
    ...     dilations=[1, 2, 4],
    ...     activations=["relu", "relu", "relu"],
    ...     bias=True,
    ...     dropouts=[0.1, 0.1, 0.1]
    ... )
    >>> tcn(data)
    """

    @classmethod
    def from_setting(cls, setting: TCNSetting) -> TCN:
        """Create TCN from TCNSetting instance

        Args:
            setting (TCNSetting): setting object for TCN

        Returns:
            TCN model
        """
        return TCN(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "TCN"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        nodes: list[int],
        kernel_sizes: list[int],
        dilations: list[int],
        activations: list[str],
        bias: bool = True,
        dropouts: list[float] | None = None,
    ) -> None:
        super().__init__()

        self._convs = ExtendedConv1DList(
            nodes=nodes,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            activations=activations,
            bias=bias,
            dropouts=dropouts,
        )

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
        """forward function which overload torch.nn.Module

        Args:
            data: IPhlowerTensorCollections
                data which receives from predecessors
            field_data: ISimulationField | None
                Constant information through training or prediction

        Returns:
            PhlowerTensor:
                Tensor object
        """

        target = data.unique_item()
        if not target.is_time_series:
            raise ValueError("Input tensor to TCN is not time-series tensor.")

        pattern = target.shape_pattern
        original_pattern = pattern.get_pattern()

        _space = pattern.get_space_pattern()
        _rank_dims = pattern.get_feature_pattern(drop_last=True)
        to_pattern = (
            f"({_space} {_rank_dims}) "
            f"{pattern.feature_pattern_symbol} {pattern.time_series_pattern}"
        )

        # To use replication pad, convert 3D tensor
        # https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad2d.html#torch.nn.ReplicationPad2d
        pattern_to_size = pattern.get_pattern_to_size(drop_last=True)
        time_last_tensor = target.rearrange(
            f"{original_pattern} -> {to_pattern}", **pattern_to_size
        )

        result = self._convs.forward(time_last_tensor)

        return result.rearrange(
            f"{to_pattern} -> {original_pattern}", **pattern_to_size
        )
