import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from phlower._base import PhlowerTensor, phlower_tensor
from phlower.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.nn._interface_iteration_solver import IOptimizeProblem
from phlower.nn._iteration_solvers._barzilai_borwein_solver import (
    AlphaCalculator,
    BarzilaiBorweinSolver,
)
from phlower.settings._nonlinear_solver_setting import (
    BarzilaiBoweinSolverSetting,
)


class QuadraticProblem(IOptimizeProblem):
    """
    f(x) = 1/2 x^T A x - b^T x
    """

    def __init__(self) -> None:
        self._A = phlower_tensor(
            tensor=np.array([[3.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        )
        self._b = phlower_tensor(
            tensor=np.array([[2.0], [5.0]], dtype=np.float32)
        )

    def desired_solution(self) -> PhlowerTensor:
        return torch.linalg.inv(self._A) @ self._b

    def step_forward(
        self, h: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        return phlower_tensor_collection({})

    def objective(
        self, value: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        x = value["x"]
        xT = x.transpose(1, 0)
        bT = self._b.transpose(1, 0)

        h = 0.5 * (xT @ self._A @ x) - bT @ x

        return phlower_tensor_collection({"x": h})

    def gradient(
        self,
        value: IPhlowerTensorCollections,
        target_keys: list[str],
        operator_keys: list[str] | None = None,
    ) -> IPhlowerTensorCollections:
        x = value["x"]
        h = self._A @ x - self._b

        return phlower_tensor_collection({"x": h})


@pytest.mark.parametrize("bb_type", ["long", "short"])
@pytest.mark.parametrize(
    "array",
    [
        ([[5.0], [1.0]]),
        ([[-4.1], [-1.0]]),
        ([[3.1], [-10.0]]),
        ([[-12.0], [110.0]]),
    ],
)
def test__can_converge_quadratic_equation(
    bb_type: str, array: list[list[float]]
):
    solver = BarzilaiBorweinSolver(
        max_iterations=10000,
        convergence_threshold=0.00001,
        divergence_threshold=100000,
        update_keys=["x"],
        bb_type=bb_type,
    )

    problem = QuadraticProblem()
    inputs = phlower_tensor_collection({"x": phlower_tensor(array)})
    h = solver.run(inputs, problem)

    actual = h.unique_item()
    desired = problem.desired_solution()
    np.testing.assert_array_almost_equal(actual, desired, decimal=5)

    # Make sure that the number of iterations is not necessary for bb method
    assert solver.get_n_iterated() < 30


# region Test for AlphaCalculator


@pytest.mark.parametrize(
    "component_wise, delta_x_shape, delta_g_shape, desired",
    [
        (True, (2, 5, 1), (2, 5, 1), (1,)),
        (True, (3, 10), (3, 10), (10,)),
        (True, (1, 1), (1, 1), (1,)),
        (False, (2, 5, 1), (2, 5, 1), ()),
        (False, (3, 10), (3, 10), ()),
        (False, (1, 1), (1, 1), ()),
    ],
)
@given(bb_type=st.sampled_from(["long", "short"]))
def test__shape_of_alpha_with_component_wise(
    component_wise: bool,
    delta_x_shape: tuple[int, ...],
    delta_g_shape: tuple[int, ...],
    desired: tuple[int, ...],
    bb_type: str,
):
    alpha_calculator = AlphaCalculator(
        component_wise=component_wise, bb_type=bb_type
    )

    delta_x = phlower_tensor(torch.rand(delta_x_shape), dtype=torch.float32)
    delta_g = phlower_tensor(torch.rand(delta_g_shape), dtype=torch.float32)

    alpha = alpha_calculator.calculate(delta_x, delta_g)
    assert alpha.shape == desired


@pytest.mark.parametrize(
    "bb_type, delta_x, delta_g, desired",
    [
        ("long", [[-1.0], [1.0], [0.0]], [[1.0], [1.0], [0.0]], 1.0),
        ("long", [[-2.0], [5.0], [0.0]], [[5.0], [2.0], [0.0]], 1.0),
        ("short", [[-2.0], [5.0], [0.0]], [[0.0], [0.0], [0.0]], 1.0),
    ],
)
@given(component_wise=st.booleans())
def test__hanlding_denominator_of_alpha(
    bb_type: str,
    delta_x: list[list[float]],
    delta_g: list[list[float]],
    desired: float,
    component_wise: bool,
):
    alpha_calculator = AlphaCalculator(
        component_wise=component_wise, bb_type=bb_type
    )

    delta_x = phlower_tensor(torch.tensor(delta_x), dtype=torch.float32)
    delta_g = phlower_tensor(torch.tensor(delta_g), dtype=torch.float32)

    alpha = alpha_calculator.calculate(delta_x, delta_g)
    assert alpha == desired


@pytest.mark.parametrize(
    "bb_type, delta_x, delta_g, not_desired",
    [
        ("long", [[-1.0], [1.0], [0.0]], [[2.0], [1.0], [0.0]], 1.0),
        ("long", [[-1.0], [1.0], [-10.0]], [[2.0], [1.0], [1.0]], 1.0),
    ],
)
@given(component_wise=st.booleans())
def test__not_hanlding_denominator_of_alpha(
    bb_type: str,
    delta_x: list[list[float]],
    delta_g: list[list[float]],
    not_desired: float,
    component_wise: bool,
):
    alpha_calculator = AlphaCalculator(
        component_wise=component_wise, bb_type=bb_type
    )

    delta_x = phlower_tensor(torch.tensor(delta_x), dtype=torch.float32)
    delta_g = phlower_tensor(torch.tensor(delta_g), dtype=torch.float32)

    alpha = alpha_calculator.calculate(delta_x, delta_g)
    assert alpha != not_desired


# endregion


@pytest.mark.parametrize("bb_type", ["long", "short"])
@pytest.mark.parametrize(
    "target_keys, update_keys, operator_keys, expected_operator_keys",
    [
        (["a"], None, None, ["a"]),
        (None, ["a"], ["ga"], ["ga"]),
        (None, ["a", "b"], ["ga", "gb"], ["ga", "gb"]),
        (None, ["a", "b"], [], []),
    ],
)
def test__initialize_from_setting(
    bb_type: str,
    target_keys: list[str] | None,
    update_keys: list[str] | None,
    operator_keys: list[str] | None,
    expected_operator_keys: list[str],
):
    setting = BarzilaiBoweinSolverSetting(
        convergence_threshold=0.01,
        max_iterations=100,
        divergence_threshold=100,
        bb_type=bb_type,
        target_keys=target_keys or [],
        update_keys=update_keys or [],
        operator_keys=operator_keys or [],
    )

    solver = BarzilaiBorweinSolver.from_setting(setting)

    assert solver._operator_keys == expected_operator_keys
    assert solver._keys == update_keys or target_keys
    assert solver._convergence_threshold == 0.01
    assert solver._max_iterations == 100
    assert solver._divergence_threshold == 100
