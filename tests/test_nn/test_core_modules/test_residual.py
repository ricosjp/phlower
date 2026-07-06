from collections.abc import Callable

import pytest
import torch
from phlower_tensor import PhlowerTensor
from phlower_tensor.collections import (
    phlower_tensor_collection,
)

from phlower.nn import Residual
from phlower.settings._module_settings import ResidualSetting
from phlower.utils import create_simulation_field


def test__can_call_parameters():
    model = Residual(
        symbols_from_input=["x", "y", "u"],
        symbols_from_field=["c"],
        equation="Diff(u, x) + Diff(u, y) - c",
    )

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "inputs, field, equation",
    [
        (
            ["x", "y", "u"],
            ["c"],
            "Diff(u, x) + Diff(u, y) - c",
        ),
        (
            ["x", "y", "z", "u"],
            ["c", "d"],
            "Diff(u, x) + Diff(u, y) + Diff(u, z) - c - d",
        ),
    ],
)
def test__parse_from_setting(
    inputs: list[str],
    field: list[str],
    equation: str,
):
    setting = ResidualSetting(
        nodes=[-1, 1],
        symbols_from_input=inputs,
        symbols_from_field=field,
        equation=equation,
    )

    model = Residual.from_setting(setting)

    assert model._input_symbols == inputs
    assert model._field_symbols == field
    assert model._equation == equation


@pytest.mark.parametrize(
    "u_func, v_func, equation, expected_func",
    [
        (
            lambda x, y: x**2 + y**2,
            lambda x, y: x**3 + y,
            "Diff(u, x) + Diff(v, x) + 10",
            lambda x, y: 2 * x + 3 * x**2 + 10,
        ),
        (
            lambda x, y: x**2 + y**2,
            lambda x, y: x**3 + y,
            "Diff(u, y) + Diff(v, y) + 5.0",
            lambda x, y: 2 * y + 1 + 5.0,
        ),
        (
            lambda x, y: x**2 + y**2,
            lambda x, y: x**3 + y,
            "Diff(u, x) + Diff(v, y)",
            lambda x, y: 2 * x + 1,
        ),
        (
            lambda x, y: x**2 + y**2,
            lambda x, y: x**3 + y,
            "Diff(u, x) + Diff(v, y) + Diff(u, y) + Diff(v, x)",
            lambda x, y: 2 * x + 1 + 2 * y + 3 * x**2,
        ),
        (
            lambda x, y: x * y**2,
            lambda x, y: (x * y) ** 2,
            "Diff(Diff(u, x), y) + Diff(Diff(v, y), x) + 1",
            lambda x, y: 2 * y + 4 * x * y + 1,
        ),
        (
            lambda x, y: x**3 + y**2,
            lambda x, y: x**3 + y**3,
            "Diff(Diff(u, x), x) + Diff(Diff(v, y), y)",
            lambda x, y: 6 * x + 6 * y,
        ),
    ],
)
def test__compute_residual(
    u_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    v_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    equation: str,
    expected_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
):
    input_symbols = ["x", "y", "u", "v"]
    _setting = ResidualSetting(
        symbols_from_input=input_symbols,
        symbols_from_field=[],
        equation=equation,
        nodes=[-1, 1],
    )
    calculator = Residual.from_setting(_setting)

    data = phlower_tensor_collection(
        {
            "x": torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
            "y": torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
        }
    )

    with torch.enable_grad():
        u = u_func(data["x"], data["y"])
        v = v_func(data["x"], data["y"])

    updated = data | phlower_tensor_collection({"u": u, "v": v})
    actual = calculator(updated, field_data=None)

    expected = expected_func(data["x"], data["y"])

    assert isinstance(actual, PhlowerTensor)
    assert isinstance(expected, PhlowerTensor)
    assert torch.allclose(actual.to_tensor(), expected.to_tensor())
    assert actual.dimension == expected.dimension


@pytest.mark.parametrize(
    "u_func, v_func, Re_value, equation, expected_func",
    [
        (
            lambda x, y: x**2 + y**2,
            lambda x, y: x**3 + y,
            100,
            "(Diff(u, x) + Diff(v, x)) / Re",
            lambda x, y: (2 * x + 3 * x**2) / 100.0,
        ),
        (
            lambda x, y: x**2 + y**2,
            lambda x, y: x**3 + y,
            50,
            "Diff(u, y) + Diff(v, y) + 5.0 / Re",
            lambda x, y: 2 * y + 1 + 5.0 / 50.0,
        ),
    ],
)
def test__compute_residual_with_constants(
    u_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    v_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    Re_value: float,
    equation: str,
    expected_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
):
    input_symbols = ["x", "y", "u", "v"]
    constants = ["Re"]
    _setting = ResidualSetting(
        symbols_from_input=input_symbols,
        symbols_from_field=constants,
        equation=equation,
        nodes=[-1, 1],
    )
    calculator = Residual.from_setting(_setting)

    data = phlower_tensor_collection(
        {
            "x": torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
            "y": torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
        }
    )

    u = u_func(data["x"], data["y"])
    v = v_func(data["x"], data["y"])

    field_data = create_simulation_field(
        {
            "Re": torch.tensor([Re_value], requires_grad=False),
        },
        None,
    )

    updated = data | phlower_tensor_collection({"u": u, "v": v})
    actual = calculator(updated, field_data=field_data)

    expected = expected_func(data["x"], data["y"])

    assert isinstance(actual, PhlowerTensor)
    assert isinstance(expected, PhlowerTensor)
    assert torch.allclose(actual.to_tensor(), expected.to_tensor())
    assert actual.dimension == expected.dimension
