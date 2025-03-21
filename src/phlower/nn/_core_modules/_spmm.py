from __future__ import annotations

import torch

from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._functionals import _functions
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import SPMMSetting


class SPMM(torch.nn.Module, IPhlowerCoreModule):
    """
    Sparse Matrix Multiplication Network

    Parameters
    ----------
    factor: float
        Factor to multiply the support matrix.
    support_name: str
        Name of the support matrix.
    transpose: bool
        Whether to transpose the support matrix.

    Examples
    --------
    >>> spmm = SPMM(factor=1.0, support_name="support", transpose=False)
    >>> spmm(data)
    """

    @classmethod
    def from_setting(cls, setting: SPMMSetting) -> SPMM:
        """Create SPMM from SPMMSetting instance

        Args:
            setting (SPMMSetting): setting object for SPMM

        Returns:
            SPMM model
        """
        return SPMM(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "SPMM"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        factor: float,
        support_name: str,
        transpose: bool = False,
        **kwards,
    ) -> None:
        super().__init__()

        self._factor = factor
        self._support_name = support_name
        self._transpose = transpose
        self._repeat = 1

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None: ...

    def get_reference_name(self) -> str | None:
        return None

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField,
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
        support = field_data[self._support_name]
        h = self._propagate(data.unique_item(), support)
        return h

    def _propagate(
        self, x: PhlowerTensor, support: PhlowerTensor
    ) -> PhlowerTensor:
        if self._transpose:
            support = torch.transpose(support, 0, 1)
        h = _functions.spmm(support * self._factor, x, repeat=self._repeat)
        return h
