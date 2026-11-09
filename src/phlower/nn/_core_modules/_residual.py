from __future__ import annotations

import abc
import operator
from collections.abc import Callable
from functools import reduce
from typing import Any, Literal, Self

import sympy as sm
import torch
from phlower_tensor import ISimulationField, PhlowerTensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

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
    symbols_from_module: list[str], optional
        List of symbols corresponding to the module outputs.
    differentiation_backend: Literal["torch.autograd", "torch.func"], optional
        The backend to use for differentiation.

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
        symbols_from_module: list[str] | None = None,
        differentiation_backend: Literal[
            "torch.autograd", "torch.func"
        ] = "torch.autograd",
    ):
        super().__init__()
        self._nodes = nodes

        self._evaluator = create_evaluator(
            backend=differentiation_backend,
            symbols_from_input=symbols_from_input,
            symbols_from_field=symbols_from_field,
            symbols_from_module=symbols_from_module or [],
            equation=equation,
        )

        self._parsed_function: Callable | None = None

    @property
    def all_symbols(self) -> list[str]:
        return self._evaluator.all_symbols

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None:
        self._evaluator.compile(parent)
        return

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

        return self._evaluator.evaluate(
            data,
            field_data=field_data,
            **kwards,
        )


class IEquationEvaluator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(
        self,
        symbols_from_input: list[str],
        symbols_from_field: list[str],
        symbols_from_module: list[str],
        equation: str,
    ): ...

    @property
    @abc.abstractmethod
    def all_symbols(self) -> list[str]: ...

    @abc.abstractmethod
    def compile(
        self, parent: IReadonlyReferenceGroup
    ) -> Callable[..., PhlowerTensor]: ...

    @abc.abstractmethod
    def evaluate(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None = None,
        **kwards,
    ) -> PhlowerTensor: ...


def create_evaluator(
    backend: str,
    symbols_from_input: list[str],
    symbols_from_field: list[str],
    symbols_from_module: list[str],
    equation: str,
) -> IEquationEvaluator:
    match backend:
        case "torch.autograd":
            return AutogradEquationEvaluator(
                symbols_from_input,
                symbols_from_field,
                symbols_from_module,
                equation,
            )
        case "torch.func":
            return TorchFuncEquationEvaluator(
                symbols_from_input,
                symbols_from_field,
                symbols_from_module,
                equation,
            )
        case _:
            raise NotImplementedError(f"Unsupported backend: {backend}")


class AutogradEquationEvaluator(IEquationEvaluator):
    def __init__(
        self,
        symbols_from_input: list[str],
        symbols_from_field: list[str],
        symbols_from_module: list[str],
        equation: str,
    ):
        self._input_symbols = symbols_from_input
        self._field_symbols = symbols_from_field
        self._module_symbols = symbols_from_module

        self._equation = equation
        self._parsed_function = None

        if self._module_symbols:
            raise ValueError(
                "Module symbols are not allowed in AutogradEquationEvaluator."
            )

    @property
    def all_symbols(self) -> list[str]:
        return self._input_symbols + self._field_symbols + self._module_symbols

    def compile(self, parent: IReadonlyReferenceGroup) -> None:
        expr = parse_equation(self._equation, self.all_symbols)
        self._parsed_function = sm.lambdify(
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

    def evaluate(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None = None,
        **kwards,
    ) -> PhlowerTensor:
        inputs = [data[s] for s in self._input_symbols]

        if self._field_symbols:
            if field_data is None:
                raise ValueError(
                    "Field data is required for computing residuals "
                    "with constants."
                )
            inputs.extend([field_data[s] for s in self._field_symbols])

        return self._parsed_function(*inputs)


class TorchFuncEquationEvaluator(IEquationEvaluator):
    def __init__(
        self,
        symbols_from_input: list[str],
        symbols_from_field: list[str],
        symbols_from_module: list[str],
        equation: str,
    ):
        self._input_symbols = symbols_from_input
        self._field_symbols = symbols_from_field
        self._module_symbols = symbols_from_module

        self._equation = equation
        self._parsed_function = None

    @property
    def all_symbols(self) -> list[str]:
        return self._input_symbols + self._field_symbols + self._module_symbols

    def compile(self, parent: IReadonlyReferenceGroup) -> None:
        _modules = {
            name: parent.search_module(name, return_adaptor=True)
            for name in self._module_symbols
        }

        expr = parse_equation(self._equation, self.all_symbols)
        self._parsed_function = _compile_expr(
            expr, self._input_symbols, self._field_symbols, _modules
        )

    def evaluate(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None = None,
        **kwards,
    ) -> PhlowerTensor:
        return self._parsed_function(data, field_data=field_data, **kwards)


def _differentiate(fn: Callable, respect_to: sm.Basic) -> Callable:
    def derivative(
        args: IPhlowerTensorCollections,
        **kwards: Any,  # noqa: ANN401
    ) -> PhlowerTensor:

        _name = str(respect_to)
        _respect_to = args[_name]

        def _warp_forward(raw_arg: torch.Tensor) -> torch.Tensor:
            _args = args | phlower_tensor_collection(
                {
                    _name: phlower_tensor(
                        raw_arg,
                        is_time_series=_respect_to.is_time_series,
                        is_voxel=_respect_to.is_voxel,
                        dimension=_respect_to.dimension,
                    )
                }
            )
            if hasattr(fn, "input_keys"):
                _args = _args.mask(fn.input_keys)
            val = fn(_args, **kwards)

            if isinstance(val, IPhlowerTensorCollections):
                # It is possible when fn is GroupModule
                val = val.unique_item()

            return val.to_tensor()

        value, vjpfunc = torch.func.vjp(_warp_forward, _respect_to.to_tensor())
        (gradient,) = vjpfunc(torch.ones_like(value))

        _value = fn(args, **kwards)
        if isinstance(_value, IPhlowerTensorCollections):
            # It is possible when fn is GroupModule
            _value = _value.unique_item()

        return phlower_tensor(
            gradient,
            is_time_series=_value.is_time_series,
            is_voxel=_value.is_voxel,
            dimension=_value.dimension,
        )

    return derivative


def _compile_expr(
    expr: sm.Expr, input_variables: list, field_variables: list, modules: dict
) -> Callable:
    """
    Compile a symbolic expression into a callable function.
    The difference with standard compilation using torch.autograd is that
    this function are lazy function and it is compatible with torch.func.vmap.

    all returned functions are expected to receive IPhlowerTensorCollections
    as the first argument and return a PhlowerTensor.
    """

    if str(expr) in modules:
        return modules[str(expr)]

    if str(expr) in input_variables:
        return lambda *args, **kwards: args[str(expr)]

    if str(expr) in field_variables:
        return lambda *args, **kwards: kwards["field_data"].get(str(expr))

    if expr.is_Number:
        value = float(expr)
        return lambda *args, **kwards: value

    if expr.func.__name__ == "Diff":
        operand, variable = expr.args
        fn = _compile_expr(operand, input_variables, field_variables, modules)
        return lambda *args, **kwards: _differentiate(fn, variable)(
            *args, **kwards
        )

    operators = {
        sm.Add: operator.add,
        sm.Mul: operator.mul,
        sm.Pow: operator.pow,
    }
    if expr.func in operators:
        op = operators[expr.func]
        children = [
            _compile_expr(child, input_variables, field_variables, modules)
            for child in expr.args
        ]
        return lambda *args, **kwards: reduce(
            op, (child(*args, **kwards) for child in children)
        )

    raise NotImplementedError(f"Unsupported expression: {expr}")
