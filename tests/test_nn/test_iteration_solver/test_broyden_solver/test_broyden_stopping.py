"""Regression tests for Broyden stopping and final-update semantics."""

from __future__ import annotations

import pytest
import torch
from phlower_tensor import phlower_tensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn._interface_iteration_solver import IOptimizeProblem
from phlower.nn._iteration_solvers._broyden_solver import BroydenSolver


class AffineResidualProblem(IOptimizeProblem):
    """One-dimensional ``R(x, theta) = a*x - theta`` problem."""

    def __init__(self, a: float, theta: torch.Tensor) -> None:
        self._a = a
        self._theta = theta
        self.calls = 0

    def step_forward(
        self, h: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        return phlower_tensor_collection({})

    def gradient(
        self,
        h: IPhlowerTensorCollections,
        update_keys: list[str],
        operator_keys: list[str] | None = None,
    ) -> IPhlowerTensorCollections:
        self.calls += 1
        x = h["x"].to_tensor()
        return phlower_tensor_collection(
            {"x": phlower_tensor(self._a * x - self._theta)}
        )


def _run(
    problem: AffineResidualProblem,
    *,
    max_iterations: int,
    convergence_threshold: float,
    divergence_threshold: float,
    exit_before_update_when_diverged: bool = False,
) -> tuple[BroydenSolver, torch.Tensor, torch.Tensor]:
    solver = BroydenSolver(
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        divergence_threshold=divergence_threshold,
        update_keys=["x"],
        exit_before_update_when_diverged=exit_before_update_when_diverged,
    )
    initial = phlower_tensor_collection(
        {"x": phlower_tensor(problem._theta * 0.0)}
    )
    result = solver.run(initial, problem)["x"].to_tensor()
    (gradient,) = torch.autograd.grad(result.sum(), problem._theta)
    return solver, result, gradient


@pytest.mark.parametrize(
    ("exit_before_update_when_diverged", "expected_value", "expected_grad"),
    [(True, 0.0, 0.0), (False, 1.0, 1.0)],
)
def test_finite_divergence_on_first_iteration_uses_flag(
    exit_before_update_when_diverged: bool,
    expected_value: float,
    expected_grad: float,
) -> None:
    theta = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    problem = AffineResidualProblem(a=3.0, theta=theta)
    solver, result, gradient = _run(
        problem,
        max_iterations=4,
        convergence_threshold=1e-9,
        divergence_threshold=0.5,
        exit_before_update_when_diverged=exit_before_update_when_diverged,
    )

    expected = torch.tensor([expected_value], dtype=result.dtype)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(
        gradient, torch.tensor([expected_grad], dtype=gradient.dtype)
    )
    assert problem.calls == 1
    assert solver.get_n_iterated() == 1
    assert solver._iterated_state.is_diverged
    assert not solver.get_converged()


@pytest.mark.parametrize(
    "exit_before_update_when_diverged",
    [True, False],
)
def test_finite_divergence_after_history_has_expected_value_and_gradient(
    exit_before_update_when_diverged: bool,
) -> None:
    theta = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    problem = AffineResidualProblem(a=3.0, theta=theta)
    solver, result, gradient = _run(
        problem,
        max_iterations=4,
        convergence_threshold=1e-9,
        divergence_threshold=1.5,
        exit_before_update_when_diverged=exit_before_update_when_diverged,
    )

    # R(0)=-theta gives x_1=theta.  The retained pair is
    # (s,y)=(theta,3*theta), so the final update is x_2=theta/3.
    expected_value = 1.0 if exit_before_update_when_diverged else 1.0 / 3.0
    expected = torch.tensor([expected_value], dtype=result.dtype)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(
        gradient, torch.tensor([expected_value], dtype=gradient.dtype)
    )
    assert problem.calls == 2
    assert solver.get_n_iterated() == 2
    assert solver._iterated_state.is_diverged
    assert not solver.get_converged()


@pytest.mark.parametrize("termination", ["converged", "max_iterations"])
@pytest.mark.parametrize("exit_before_update_when_diverged", [False, True])
def test_diverged_only_flag_does_not_change_other_stops(
    termination: str,
    exit_before_update_when_diverged: bool,
) -> None:
    theta = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    problem = AffineResidualProblem(a=0.5, theta=theta)
    solver, result, gradient = _run(
        problem,
        max_iterations=2,
        convergence_threshold=0.75 if termination == "converged" else 1e-9,
        divergence_threshold=10.0,
        exit_before_update_when_diverged=exit_before_update_when_diverged,
    )

    # The first update is x_1=theta.  The second residual is -theta/2,
    # giving x_2=2*theta, regardless of the divergence-only flag.
    expected = 2.0
    expected_tensor = torch.tensor([expected], dtype=result.dtype)
    torch.testing.assert_close(result, expected_tensor)
    torch.testing.assert_close(
        gradient, torch.tensor([expected], dtype=gradient.dtype)
    )
    assert problem.calls == 2
    assert solver.get_n_iterated() == 2
    assert solver._iterated_state.is_converged is (termination == "converged")
    assert not solver._iterated_state.is_diverged
