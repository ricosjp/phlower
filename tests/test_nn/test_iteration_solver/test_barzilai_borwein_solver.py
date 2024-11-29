import numpy as np
import pytest
import torch
from phlower._base import PhlowerTensor, phlower_tensor
from phlower.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.nn._interface_iteration_solver import IOptimizeProblem
from phlower.nn._iteration_solvers._barzilai_borwein_solver import (
    BarzilaiBorweinSolver,
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

    def objective(
        self, value: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        x = value["x"]
        xT = x.transpose(1, 0)
        bT = self._b.transpose(1, 0)

        h = 0.5 * (xT @ self._A @ x) - bT @ x

        return phlower_tensor_collection({"x": h})

    def gradient(
        self, value: IPhlowerTensorCollections
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
        divergence_threshold=0.00001,
        keys=["x"],
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
