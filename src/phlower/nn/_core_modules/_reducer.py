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

        _REGISTERED_OPERATORS = {
            "add": torch.add,
            "mul": torch.mul
        }

        self._operator = _REGISTERED_OPERATORS[self._operator_name]


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
            supports (dict[str, PhlowerTensor], optional):
                Graph object. Defaults to None. Reducer will not use it.

        Returns:
            PhlowerTensor: Tensor object
        """

        tensors = tuple(data.values())
        ans = tensors[0]
        for i in range(len(tensors)-1):
            ans = self._operator(ans, tensors[i+1])

        return self._activation_func(ans)
