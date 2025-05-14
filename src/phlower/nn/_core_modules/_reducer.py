from __future__ import annotations

from functools import reduce

import torch
from typing_extensions import Self

from phlower._base._functionals import broadcast_to
from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _utils
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import ReducerSetting


class Reducer(IPhlowerCoreModule, torch.nn.Module):
    """Reducer is a neural network module that performs a reduction operation
    on the input tensor.

    Parameters
    ----------
    activation: str
        Name of the activation function to apply to the output.
    operator: str
        Name of the operator to apply to the input tensors. "add" or "mul".
        Default is "add".
    nodes: list[int]
        List of feature dimension sizes (The last value of tensor shape).

    Examples
    --------
    >>> reducer = Reducer(activation="relu", operator="add", nodes=[10, 20, 30])
    >>> reducer(data)
    """

    _REGISTERED_OPERATORS = {"add": torch.add, "mul": torch.mul}

    @classmethod
    def from_setting(cls, setting: ReducerSetting) -> Self:
        """Create Reducer from setting object

        Args:
            setting: ReducerSetting
                setting object

        Returns:
            Self: Reducer
        """
        return Reducer(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of Reducer

        Returns:
            str: name
        """
        return "Reducer"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(self, activation: str, operator: str, nodes: list[int] = None):
        super().__init__()
        self._nodes = nodes
        self._activation_name = activation
        self._activation_func = _utils.ActivationSelector.select(activation)
        self._operator_name = operator

        self._operator = self._REGISTERED_OPERATORS[self._operator_name]

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

        tensors = tuple(data.values())
        ans = tensors[0]

        ans = reduce(
            lambda x, y: self._operator(*_to_broadcastable_shape(x, y)),
            tensors,
        )

        return self._activation_func(ans)


def _to_broadcastable_shape(
    tensor1: PhlowerTensor, tensor2: PhlowerTensor
) -> tuple[PhlowerTensor, PhlowerTensor]:
    if len(tensor1.shape) == len(tensor2.shape):
        return tensor1, tensor2

    tensor_s, tensor_l = sorted([tensor1, tensor2], key=lambda x: len(x.shape))

    tensor_s = broadcast_to(tensor_s, tensor_l.shape_pattern)
    return tensor_s, tensor_l
