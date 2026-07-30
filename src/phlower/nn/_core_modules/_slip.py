from __future__ import annotations

import torch
from phlower_tensor import ISimulationField, PhlowerTensor
from phlower_tensor.collections import IPhlowerTensorCollections
from phlower_tensor.functionals import einsum

from phlower.nn._core_modules import _utils
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import SlipSetting


class Slip(IPhlowerCoreModule, torch.nn.Module):
    """Slip is a neural network module that applies the slip
    boundary condition ``u <- u - (u . n) n`` on nodes where the flag
    field is one.

    The three inputs are identified by name:
    the surface normal ``(N, d, 1)`` and the flag ``(N, 1)``,
    the remaining input being the value ``(N, d, F)``, a rank-1 tensor.
    Each input may have a time-series axis (first axis).

    Parameters
    ----------
    activation: str
        Name of the activation function to apply to the output.
    normal_name: str
        Name of the surface normal field.
    flag_name: str
        Name of the flag field, which is one on slip boundary nodes and
        zero elsewhere.
    nodes: list[int] | None (optional)
        List of feature dimension sizes (The last value of tensor shape).
        Defaults to None.

    Examples
    --------
    >>> slip = Slip(
    ...     activation="identity",
    ...     normal_name="normal",
    ...     flag_name="slip_flag",
    ... )
    >>> slip(data)

    """

    @classmethod
    def from_setting(cls, setting: SlipSetting) -> Slip:
        """Create Slip from setting object

        Args:
            setting (SlipSetting): setting object

        Returns:
            Self: Slip
        """
        return Slip(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of Slip

        Returns:
            str: name
        """
        return "Slip"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        activation: str,
        normal_name: str,
        flag_name: str,
        nodes: list[int] = None,
    ):
        super().__init__()
        self._nodes = nodes
        self._activation_name = activation
        self._activation_func = _utils.ActivationSelector.select(activation)
        self._normal_name = normal_name
        self._flag_name = flag_name

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
        normal = data.pop(self._normal_name)
        flag = data.pop(self._flag_name)
        value = data.unique_item()
        if value.rank() != 1:
            raise ValueError(
                "Slip is only applicable to a rank-1 tensor. "
                f"actual rank: {value.rank()}"
            )
        if normal.shape[-1] != 1 or flag.shape[-1] != 1:
            raise ValueError(
                "feature dimension of normal and flag must be 1 in Slip. "
                f"actual: normal {normal.shape[-1]}, flag {flag.shape[-1]}"
            )

        equation = self._create_equation(value, normal, flag)
        normal_component = einsum(
            equation, value, normal, normal, flag, dimension=value.dimension
        )
        return self._activation_func(value - normal_component)

    def _create_equation(
        self,
        value: PhlowerTensor,
        normal: PhlowerTensor,
        flag: PhlowerTensor,
    ) -> str:
        """Create the einsum equation computing ``(u . n) n * flag``,
        e.g. ``npf,nph,nqh,ng->nqf``:
        ``n`` is nodes,
        ``p``/``q`` are spatial components,
        ``f`` is the value's feature axis,
        and ``h``/``g`` are the size-1 feature axes of normal and flag,
        which broadcast over ``f``.
        Each operand keeps a leading ``t`` only if it is a time series.
        """
        space = value.shape_pattern.get_space_pattern(omit_space=True)
        time_value = value.shape_pattern.time_series_pattern
        time_normal = normal.shape_pattern.time_series_pattern
        time_flag = flag.shape_pattern.time_series_pattern
        is_time_series = (
            value.is_time_series or normal.is_time_series or flag.is_time_series
        )
        time_result = "t" if is_time_series else ""
        return (
            f"{time_value}{space}pf,{time_normal}{space}ph,"
            f"{time_normal}{space}qh,{time_flag}{space}g"
            f"->{time_result}{space}qf"
        )
