from __future__ import annotations

from collections.abc import Callable
from typing import Self

import sympy as sm
import torch
from phlower_tensor import ISimulationField, PhlowerTensor
from phlower_tensor.collections import IPhlowerTensorCollections

from phlower import phlower_tensor
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import ResidualSetting
from phlower.utils import calculate_grad, parse_equation


class Residual(IPhlowerCoreModule, torch.nn.Module):
    """
    Residual module computes the residual of a given equation
    based on the input data and field data.

    Parameters
    ----------
    symbols_from_inputs: list[str]
        List of symbols corresponding to the input data.
    symbols_from_field: list[str]
        List of symbols corresponding to the field data.
    equation: str
        The equation to compute the residual.
    nodes: list[int], optional
        List of feature dimension sizes (The last value of tensor shape).

    Examples
    --------
    >>> residual = Residual(
    >>>     symbols_from_inputs=["x", "y"],
    >>>     symbols_from_field=["c"],
    >>>     equation="Diff(u, x) + Diff(u, y) - c",
    >>> )
    >>> residual(data, field_data)
    """

    @classmethod
    def from_setting(cls, setting: ResidualSetting) -> Self:
        return Residual(**setting.model_dump())

    @classmethod
    def get_nn_name(cls) -> str:
        return "Residual"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        symbols_from_input: list[str],
        symbols_from_field: list[str],
        equation: str,
        nodes: list[int] = None,
    ):
        super().__init__()
        self._nodes = nodes
        self._input_symbols = symbols_from_input
        self._field_symbols = symbols_from_field
        self._equation = equation

        self._parsed_function = self._construct_equation()

    def _construct_equation(self) -> Callable[..., PhlowerTensor]:
        expr = parse_equation(self._equation, self.all_symbols)
        return sm.lambdify(
            self.all_symbols,
            expr,
            modules=[
                {
                    "Diff": lambda y, x: calculate_grad(
                        y, [x], allow_unused=True
                    )[0],
                    "Integer": lambda x: torch.tensor(x, dtype=torch.int32),
                    "Float": lambda x: torch.tensor(x, dtype=torch.float32),
                }
            ],
        )

    @property
    def all_symbols(self) -> list[str]:
        return self._input_symbols + self._field_symbols

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

        if not torch.is_grad_enabled():
            return phlower_tensor([torch.nan])

        if kwards.get("compute_residual", True) is False:
            return phlower_tensor([torch.nan])

        inputs = [data[s] for s in self._input_symbols]

        if self._field_symbols:
            if field_data is None:
                raise ValueError(
                    "Field data is required for computing residuals "
                    "with constants."
                )
            inputs.extend([field_data[s] for s in self._field_symbols])

        return self._parsed_function(*inputs)
