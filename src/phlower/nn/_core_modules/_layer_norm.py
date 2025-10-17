from __future__ import annotations

import torch
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import LayerNormSetting


class LayerNorm(IPhlowerCoreModule, torch.nn.Module):
    """LayerNorm is a neural network module that applies
    Layer Normalization on the input tensor.

    Parameters
    ----------
    nodes: list[int]
        List of feature dimension sizes (The last value of tensor shape).
    eps: float
        A value added to the denominator for numerical stability.
    elementwise_affine: bool
        Whether to learn additional affine parameters.
    bias: bool
        Whether to learn an additive bias if `elementwise_affine` is `True`.

    Examples
    --------
    >>> layer_norm = LayerNorm()
    >>> layer_norm(data)
    """

    @classmethod
    def from_setting(cls, setting: LayerNormSetting) -> Self:
        """Generate LayerNorm from setting object

        Args:
            setting (LayerNormSetting): setting object

        Returns:
            Self: LayerNorm object
        """
        return LayerNorm(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "LayerNorm"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        nodes: list[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self._nodes = nodes
        self._layer_norm = torch.nn.LayerNorm(
            nodes[0],
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
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
        """forward function which overloads torch.nn.Module

        Args:
            data: IPhlowerTensorCollections
                data which receives from predecessors
            field_data: ISimulationField | None
                Constant information through training or prediction

        Returns:
            PhlowerTensor: Tensor object
        """
        h = data.unique_item()
        h = self._layer_norm.forward(h)
        return h
