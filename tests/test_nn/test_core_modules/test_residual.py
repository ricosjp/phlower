from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from phlower_tensor import PhlowerTensor, phlower_tensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn import PhlowerGroupModule, Residual
from phlower.settings import PhlowerSetting
from phlower.settings._module_settings import ResidualSetting
from phlower.utils import create_simulation_field


class _DummyModule:
    def __init__(self, input_keys: list[str], function: Callable):
        self.input_keys = input_keys
        self.function = function

    def __call__(self, args: IPhlowerTensorCollections, **kwargs):
        args = [args[key] for key in self.input_keys]
        return self.function(*args)


def create_parent_for_torch_func_like_residual(
    setting: ResidualSetting, functions: dict[str, Callable]
) -> MagicMock | None:
    if setting.differentiation_backend != "torch.func":
        return None

    mock = MagicMock()
    mock.search_module.side_effect = lambda x: _DummyModule(
        input_keys=setting.symbols_from_input, function=functions[x]
    )
    return mock


@pytest.mark.parametrize(
    "differentiation_backend", ["torch.autograd", "torch.func"]
)
def test__can_call_parameters(differentiation_backend: str):
    model = Residual(
        symbols_from_input=["x", "y", "u"],
        symbols_from_field=["c"],
        equation="Diff(u, x) + Diff(u, y) - c",
        differentiation_backend=differentiation_backend,
    )

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "differentiation_backend", ["torch.autograd", "torch.func"]
)
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
    differentiation_backend: str,
):
    setting = ResidualSetting(
        nodes=[-1, 1],
        symbols_from_input=inputs,
        symbols_from_field=field,
        equation=equation,
        differentiation_backend=differentiation_backend,
    )

    model = Residual.from_setting(setting)
    model.resolve()


@pytest.mark.parametrize(
    "differentiation_backend", ["torch.autograd", "torch.func"]
)
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
    differentiation_backend: str,
):
    input_symbols = ["x", "y", "u", "v"]
    _setting = ResidualSetting(
        symbols_from_input=input_symbols,
        symbols_from_field=[],
        equation=equation,
        nodes=[-1, 1],
    )
    calculator = Residual.from_setting(_setting)
    parent = create_parent_for_torch_func_like_residual(
        _setting, {"u": u_func, "v": v_func}
    )
    calculator.resolve(parent=parent)

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
    "differentiation_backend", ["torch.autograd", "torch.func"]
)
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
    differentiation_backend: str,
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
    parent = create_parent_for_torch_func_like_residual(
        _setting, {"u": u_func, "v": v_func}
    )
    calculator.resolve(parent=parent)

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


def test__compare_residual():

    # ----------- autograd
    setting = PhlowerSetting.read_yaml(
        Path(__file__).parent / "data/residual/laplacian_torch_autograd.yml"
    )
    setting.model.resolve()
    model_autograd = PhlowerGroupModule.from_setting(
        setting.model.network
    ).double()

    x = torch.rand(10, 3, 1, dtype=torch.float64, requires_grad=True)
    y = torch.rand(10, 3, 1, dtype=torch.float64, requires_grad=True)
    inputs = phlower_tensor_collection(
        {"x": phlower_tensor(x), "y": phlower_tensor(y)}
    )
    autograd_laplacian = model_autograd(inputs, field_data=None)["laplacian"]
    # ------------

    # ------------ torch.func
    setting = PhlowerSetting.read_yaml(
        Path(__file__).parent / "data/residual/laplacian_torch_func.yml"
    )
    setting.model.resolve()
    model_func = PhlowerGroupModule.from_setting(setting.model.network).double()

    # Use the same weights for the MLP_U module in the functional model
    # as in the autograd model
    model_func.search_module("COMPUTE_U").search_module(
        "MLP_U"
    ).load_state_dict(model_autograd.search_module("MLP_U").state_dict())

    x = x.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    inputs = phlower_tensor_collection(
        {"x": phlower_tensor(x), "y": phlower_tensor(y)}
    )
    func_laplacian = model_func(inputs, field_data=None)["laplacian"]
    # ------------

    assert autograd_laplacian.to_tensor() is not None
    assert func_laplacian.to_tensor() is not None
    assert torch.allclose(
        autograd_laplacian.to_tensor(), func_laplacian.to_tensor()
    )
