from collections import deque

import numpy as np
import pytest
import torch
from phlower_tensor import PhlowerTensor, phlower_tensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn._interface_iteration_solver import IOptimizeProblem
from phlower.nn._iteration_solvers import _broyden_solver as broyden_module
from phlower.nn._iteration_solvers._broyden_solver import BroydenSolver
from phlower.settings._iteration_solver_setting import BroydenSolverSetting


class QuadraticProblem(IOptimizeProblem):
    """
    f(x) = 1/2 x^T A x - b^T x
    """

    def __init__(self, record_intermediate_tensor: bool = False) -> None:
        self._A = phlower_tensor(
            tensor=np.array([[3.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        )
        self._b = phlower_tensor(
            tensor=np.array([[2.0], [5.0]], dtype=np.float32)
        )
        self._recorded_intermediate_tensor = record_intermediate_tensor
        self._recorded: list[PhlowerTensor] = []

    def get_recorded(self) -> list[PhlowerTensor]:
        return self._recorded

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
        update_keys: list[str],
        operator_keys: list[str] | None = None,
    ) -> IPhlowerTensorCollections:
        x = value["x"]
        h = self._A @ x - self._b
        if self._recorded_intermediate_tensor:
            h.to_tensor().retain_grad()
            self._recorded.append(h)

        return phlower_tensor_collection({"x": h})


def test__inner_preserves_product_dimension() -> None:
    a = phlower_tensor([1.0, 2.0], dimension={"L": 1})
    b = phlower_tensor([3.0, 4.0], dimension={"L": 1})

    actual = broyden_module._inner(a, b)

    assert actual.to_tensor().item() == pytest.approx(11.0)
    assert actual.dimension == phlower_tensor(1.0, dimension={"L": 2}).dimension


def test__memory_length_2_retains_latest_solver_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingDeque(deque[tuple[PhlowerTensor, PhlowerTensor]]):
        def __init__(self, maxlen: int | None = None) -> None:
            super().__init__(maxlen=maxlen)
            self.appended: list[tuple[PhlowerTensor, PhlowerTensor]] = []
            histories.append(self)

        def append(self, pair: tuple[PhlowerTensor, PhlowerTensor]) -> None:
            self.appended.append(pair)
            super().append(pair)

    histories: list[RecordingDeque] = []
    monkeypatch.setattr(broyden_module, "deque", RecordingDeque)
    solver = BroydenSolver(
        max_iterations=10000,
        convergence_threshold=0.00001,
        divergence_threshold=100000,
        update_keys=["x"],
        memory_length=2,
    )

    problem = QuadraticProblem()
    inputs = phlower_tensor_collection(
        {"x": phlower_tensor([[-12.0], [110.0]])}
    )
    result = solver.run(inputs, problem)

    np.testing.assert_array_almost_equal(
        result.unique_item(), problem.desired_solution(), decimal=5
    )
    assert len(histories) == 1
    history = histories[0]
    assert len(history.appended) >= 3
    assert len(history) == 2
    assert [id(pair) for pair in history] == [
        id(pair) for pair in history.appended[-2:]
    ]


@pytest.mark.parametrize(
    "array",
    [
        ([[5.0], [1.0]]),
        ([[-4.1], [-1.0]]),
        ([[3.1], [-10.0]]),
        ([[-12.0], [110.0]]),
    ],
)
@pytest.mark.parametrize("memory_length", [10, 1])
def test__can_converge_quadratic_equation(
    array: list[list[float]], memory_length: int
):
    solver = BroydenSolver(
        max_iterations=10000,
        convergence_threshold=0.00001,
        divergence_threshold=100000,
        update_keys=["x"],
        memory_length=memory_length,
    )

    problem = QuadraticProblem()
    inputs = phlower_tensor_collection({"x": phlower_tensor(array)})
    h = solver.run(inputs, problem)

    actual = h.unique_item()
    desired = problem.desired_solution()
    np.testing.assert_array_almost_equal(actual, desired, decimal=5)

    assert solver.get_n_iterated() < 30


@pytest.mark.parametrize(
    "array",
    [
        ([[5.0], [1.0]]),
        ([[-4.1], [-1.0]]),
        ([[3.1], [-10.0]]),
        ([[-12.0], [110.0]]),
    ],
)
def test__can_converge_quadratic_equation_memory_length_2(
    array: list[list[float]],
):
    # Rebuild H from the retained pairs after the bounded history overflows.
    solver = BroydenSolver(
        max_iterations=10000,
        convergence_threshold=0.00001,
        divergence_threshold=100000,
        update_keys=["x"],
        memory_length=2,
    )

    problem = QuadraticProblem()
    inputs = phlower_tensor_collection({"x": phlower_tensor(array)})
    h = solver.run(inputs, problem)

    actual = h.unique_item()
    desired = problem.desired_solution()
    np.testing.assert_array_almost_equal(actual, desired, decimal=5)

    assert solver.get_n_iterated() < 30


@pytest.mark.parametrize(
    "update_keys, operator_keys, expected_operator_keys",
    [
        (["a"], ["ga"], ["ga"]),
        (["a", "b"], ["ga", "gb"], ["ga", "gb"]),
        (["a", "b"], [], []),
    ],
)
@pytest.mark.parametrize("exit_before_update_when_diverged", [True, False])
def test__initialize_from_setting(
    update_keys: list[str] | None,
    operator_keys: list[str] | None,
    expected_operator_keys: list[str],
    exit_before_update_when_diverged: bool,
):
    setting = BroydenSolverSetting(
        convergence_threshold=0.01,
        max_iterations=100,
        divergence_threshold=100,
        memory_length=10,
        update_keys=update_keys or [],
        operator_keys=operator_keys or [],
        exit_before_update_when_diverged=exit_before_update_when_diverged,
    )

    solver = BroydenSolver.from_setting(setting)

    assert solver._operator_keys == expected_operator_keys
    assert solver._keys == update_keys
    assert solver.convergence_threshold == 0.01
    assert solver.max_iterations == 100
    assert solver.divergence_threshold == 100
    assert solver._memory_length == 10
    assert (
        solver._exit_before_update_when_diverged
        == exit_before_update_when_diverged
    )
    assert "skip_last_update" not in setting.model_dump()
