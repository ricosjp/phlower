import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given
from phlower._base import phlower_tensor
from phlower.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.nn._interface_iteration_solver import IOptimizeProblem
from phlower.nn._iteration_solvers._simple_iteration_solver import (
    SimpleIterationSolver,
)


class ConvergedProblem(IOptimizeProblem):
    def __init__(self): ...

    def objective(
        self, value: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        h = value.unique_item()
        results = phlower_tensor_collection({"x": torch.sqrt(2.0 * h)})
        return results

    def gradient(
        self, value: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        return phlower_tensor_collection({})

    def desired(self, n_items: int) -> torch.Tensor:
        return torch.ones((n_items,)) * 2.0


@given(
    st.lists(
        st.floats(
            min_value=1,
            max_value=100000,
            width=32,
            allow_infinity=False,
            allow_nan=False,
        ),
        min_size=1,
    )
)
def test__can_converged(values: list[float]):
    solver = SimpleIterationSolver(
        max_iterations=10000, divergence_threshold=0.00001, target_keys=["x"]
    )
    problem = ConvergedProblem()

    x = phlower_tensor_collection({"x": phlower_tensor(values)})
    actuals = solver.run(x, problem)
    actual = actuals.unique_item()

    np.testing.assert_array_almost_equal(
        actual, problem.desired(n_items=len(values)), decimal=5
    )
