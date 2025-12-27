from __future__ import annotations

from typing import Self

import torch
from phlower_tensor import ISimulationField, PhlowerTensor
from phlower_tensor.collections import IPhlowerTensorCollections

from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import NaNToNumSetting


class NaNToNum(torch.nn.Module, IPhlowerCoreModule):
    """NaNToNum is a neural network module that replaces NaN and infinite values
    in the input tensor with specified numerical values.

    Parameters
    ----------
    nan: float
        The value to replace NaN values with. Default is 0.0.
    posinf: float | None
        The value to replace positive infinity values with. Default is None,
        which means positive infinity values are not replaced.
    neginf: float | None
        The value to replace negative infinity values with. Default is None,
        which means negative infinity values are not replaced.

    Examples
    --------
    >>> nan_to_num = NaNToNum(nan=0.0, posinf=1e6, neginf=-1e6)
    >>> nan_to_num(data)
    """

    @classmethod
    def from_setting(cls, setting: NaNToNumSetting) -> Self:
        """Generate MLP from setting object

        Args:
            setting (NaNToNumSetting): setting object

        Returns:
            Self: MLP object
        """
        return NaNToNum(**setting.model_dump())

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "NaNToNum"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        nodes: list[int] | None = None,
        nan: float = 0.0,
        posinf: float | None = None,
        neginf: float | None = None,
    ) -> None:
        super().__init__()

        self._nodes = nodes  # unused
        self._nan = nan
        self._posinf = posinf
        self._neginf = neginf

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

        h = torch.nan_to_num(
            h,
            nan=self._nan,
            posinf=self._posinf,
            neginf=self._neginf,
        )
        return h
