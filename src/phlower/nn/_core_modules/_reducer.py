from __future__ import annotations

import torch
from typing_extensions import Self

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
    """Reducer"""

    _REGISTERED_OPERATORS = {"add": torch.add, "mul": torch.mul}

    @classmethod
    def from_setting(cls, setting: ReducerSetting) -> Self:
        """Create Reducer from setting object

        Args:
            setting (ReducerSetting): setting object

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
            data (IPhlowerTensorCollections):
                data which receives from predecessors
            field_data (ISimulationField):
                Constant information through training or prediction

        Returns:
            PhlowerTensor: Tensor object
        """

        tensors = tuple(data.values())
        ans = tensors[0]
        for i in range(len(tensors) - 1):
            vs, vl = _convert_to_broadcastable_shape(ans, tensors[i + 1])
            ans = self._operator(vs, vl)

        return self._activation_func(ans)


def _convert_to_broadcastable_shape(
    tensor1: PhlowerTensor, tensor2: PhlowerTensor
) -> tuple[PhlowerTensor, PhlowerTensor]:
    if len(tensor1.shape) == len(tensor2.shape):
        return tensor1, tensor2

    tensor_s, tensor_l = sorted([tensor1, tensor2], key=lambda x: len(x.shape))

    shape_s = tensor_s.shape
    shape_l = tensor_l.shape
    new_shape: list[int] = []

    index_s = 0
    for vl in shape_l:
        if index_s >= len(shape_s):
            new_shape.append(1)
            continue

        front = shape_s[index_s]
        if front == vl:
            new_shape.append(vl)
            index_s += 1

        else:
            new_shape.append(1)

    tensor_s = tensor_s.reshape(
        tuple(new_shape),
        is_time_series=tensor_s.is_time_series,
        is_voxel=tensor_s.is_voxel,
    )
    return tensor_s, tensor_l
