import sympy as sm
import torch
from phlower_tensor import PhlowerTensor
from sympy.parsing.sympy_parser import parse_expr


def parse_equation(equation: str, symbols: list[str]) -> sm.Expr:
    # Diff is treated as special function
    _local_dict = {
        "Diff": sm.Function("Diff"),
        "Integer": sm.Integer,
        "Float": sm.Float,
    }
    # NOTE: Order of input_symbols and constants is important for lambdify
    _local_dict.update({s: sm.symbols(s) for s in symbols})

    return parse_expr(equation, local_dict=_local_dict, global_dict={})


def calculate_grad(
    target: PhlowerTensor,
    respect_to: list[PhlowerTensor],
    allow_unused: bool = False,
) -> list[PhlowerTensor]:
    assert isinstance(respect_to, list), f"Invalid type: {type(respect_to)=}"
    return [
        _calculate_grad(
            target=target,
            respect_to=var,
            allow_unused=allow_unused,
        )
        for var in respect_to
    ]


def _calculate_grad(
    target: PhlowerTensor,
    respect_to: PhlowerTensor,
    allow_unused: bool = False,
) -> PhlowerTensor:
    _tensor_value = _calculate_grad_raw(
        value=target.to_tensor(),
        respect_to=respect_to.to_tensor(),
        allow_unused=allow_unused,
    )[0]

    if target.dimension is not None:
        _dimension = target.dimension / respect_to.dimension
    else:
        assert respect_to.dimension is None, (
            f"respect_to must have no dimension. Input: {respect_to.dimension}."
        )
        _dimension = None

    return PhlowerTensor.from_pattern(
        tensor=_tensor_value,
        dimension_tensor=_dimension,
        pattern=respect_to.shape_pattern,
    )


def _calculate_grad_raw(
    value: torch.Tensor,
    respect_to: torch.Tensor,
    allow_unused: bool = False,
) -> list[torch.Tensor]:
    grad_outputs = [torch.ones_like(value)]
    gradients = torch.autograd.grad(
        [value],
        respect_to,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        allow_unused=allow_unused,
    )

    # NOTE: Uncomment the following lines
    #  if you want to handle NaN values in gradients
    _gradients = [None for _ in range(len(gradients))]
    for i, grad in enumerate(gradients):
        if grad is None:
            _gradients[i] = torch.zeros_like(respect_to)
        else:
            _gradients[i] = grad

    return _gradients
