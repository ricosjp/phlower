from collections.abc import Callable

import hypothesis.strategies as st
import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis.extra import numpy as extra_np
from phlower_tensor import phlower_tensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn._interface_iteration_solver import IOptimizeProblem
from phlower.nn._iteration_solvers import EmptySolver


class TestProblem(IOptimizeProblem):
    def __init__(self, function: Callable):
        self._function = function

    def step_forward(
        self, value: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        return self._function(value)

    def gradient(
        self, value: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        return phlower_tensor_collection({})


@pytest.mark.parametrize("function", [torch.abs, torch.relu, torch.tanh])
@given(
    extra_np.arrays(
        dtype=np.dtypes.Float32DType(),
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        ),
    )
)
def test__forward_any_function(function: Callable, values: np.ndarray):
    solver = EmptySolver()

    _tensor = phlower_tensor(values)
    _problem = TestProblem(function)
    result = solver.run(_tensor, _problem)

    np.testing.assert_array_almost_equal(result, function(_tensor))
