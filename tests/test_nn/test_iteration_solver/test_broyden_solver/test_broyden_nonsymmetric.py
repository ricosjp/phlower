"""Regression tests for Broyden solves with nonsymmetric residuals.

The residuals below need not be gradients of scalar objective functions; their
Jacobians are intentionally nonsymmetric.
"""

from __future__ import annotations

import numpy as np
import torch
from phlower_tensor import phlower_tensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn._interface_iteration_solver import IOptimizeProblem
from phlower.nn._iteration_solvers._broyden_solver import BroydenSolver


class LinearResidualProblem(IOptimizeProblem):
    """Solve ``R(x) = A x - b`` without requiring a scalar objective."""

    def __init__(self, a: torch.Tensor, b: torch.Tensor) -> None:
        self._a = a
        self._b = b

    def residual_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self._a @ x - self._b

    def step_forward(
        self, h: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        return phlower_tensor_collection({})

    def gradient(
        self,
        value: IPhlowerTensorCollections,
        update_keys: list[str],
        operator_keys: list[str] | None = None,
    ) -> IPhlowerTensorCollections:
        x = value["x"].to_tensor()
        return phlower_tensor_collection(
            {"x": phlower_tensor(self.residual_tensor(x))}
        )


class NonlinearResidualProblem(IOptimizeProblem):
    """Solve ``R(x) = A x + eps * (x * x) - b``.

    The Jacobian ``A + diag(2 * eps * x)`` is not assumed symmetric, so this
    residual is not treated as the gradient of a scalar objective.
    """

    def __init__(self, a: torch.Tensor, b: torch.Tensor, eps: float) -> None:
        self._a = a
        self._b = b
        self._eps = eps

    @classmethod
    def from_known_root(
        cls, a: torch.Tensor, x_star: torch.Tensor, eps: float
    ) -> NonlinearResidualProblem:
        b = a @ x_star + eps * (x_star * x_star)
        return cls(a, b, eps)

    def residual_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self._a @ x + self._eps * (x * x) - self._b

    def step_forward(
        self, h: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        return phlower_tensor_collection({})

    def gradient(
        self,
        value: IPhlowerTensorCollections,
        update_keys: list[str],
        operator_keys: list[str] | None = None,
    ) -> IPhlowerTensorCollections:
        x = value["x"].to_tensor()
        return phlower_tensor_collection(
            {"x": phlower_tensor(self.residual_tensor(x))}
        )


class CubicResidualProblem(NonlinearResidualProblem):
    """Solve ``R(x) = A x + eps * x**3 - b``."""

    @classmethod
    def from_known_root(
        cls, a: torch.Tensor, x_star: torch.Tensor, eps: float
    ) -> CubicResidualProblem:
        b = a @ x_star + eps * (x_star**3)
        return cls(a, b, eps)

    def residual_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self._a @ x + self._eps * (x**3) - self._b


def _run_broyden(
    problem: LinearResidualProblem | NonlinearResidualProblem,
    x0: torch.Tensor,
    *,
    max_iterations: int = 200,
) -> tuple[BroydenSolver, IPhlowerTensorCollections]:
    solver = BroydenSolver(
        max_iterations=max_iterations,
        convergence_threshold=1e-6,
        divergence_threshold=1e6,
        update_keys=["x"],
        memory_length=10,
    )
    result = solver.run(
        phlower_tensor_collection({"x": phlower_tensor(x0)}), problem
    )
    return solver, result


def _assert_broyden_result(
    solver: BroydenSolver,
    result: IPhlowerTensorCollections,
    problem: NonlinearResidualProblem | LinearResidualProblem,
    expected: torch.Tensor,
    *,
    rtol: float = 2e-4,
    atol: float = 2e-4,
) -> None:
    actual = result["x"].to_tensor()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    residual = problem.residual_tensor(actual)
    assert torch.isfinite(residual).all()
    assert torch.linalg.norm(residual).item() < 1e-5
    assert solver.get_converged()
    assert not solver._iterated_state.is_diverged
    assert 1 <= solver.get_n_iterated() < solver.max_iterations


def test_broyden_converges_nonsymmetric_linear_system() -> None:
    a = torch.tensor([[3.0, 2.0], [0.0, 4.0]], dtype=torch.float32)
    assert not torch.allclose(a, a.T)
    b = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    problem = LinearResidualProblem(a, b)
    x0 = torch.tensor([[0.0], [0.0]], dtype=torch.float32)

    solver, result = _run_broyden(problem, x0, max_iterations=100)
    expected = torch.linalg.solve(a, b)
    _assert_broyden_result(
        solver, result, problem, expected, rtol=1e-4, atol=1e-4
    )


def test_broyden_converges_2d_nonsymmetric_nonlinear_residual() -> None:
    a = torch.tensor([[0.45, 0.12], [0.04, 0.55]], dtype=torch.float32)
    assert not torch.allclose(a, a.T)
    x_star = torch.tensor([[0.8], [-0.35]], dtype=torch.float32)
    eps = 0.12
    x0 = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    problem = NonlinearResidualProblem.from_known_root(a, x_star, eps)
    solver, result = _run_broyden(problem, x0)
    _assert_broyden_result(
        solver, result, problem, x_star, rtol=2e-4, atol=2e-4
    )


def test_broyden_converges_seeded_12d_nonsymmetric_nonlinear_residual() -> None:
    rng = np.random.default_rng(3)
    n = 12
    diag = 0.35 + 0.25 * rng.random(n)
    off = 0.04 * rng.standard_normal((n, n))
    a = torch.tensor(np.diag(diag) + np.triu(off, 1), dtype=torch.float32)
    assert not torch.allclose(a, a.T)
    x_star = torch.tensor(rng.standard_normal((n, 1)).astype(np.float32) * 0.4)
    eps = 0.08
    x0 = torch.zeros((n, 1), dtype=torch.float32)
    problem = NonlinearResidualProblem.from_known_root(a, x_star, eps)

    solver, result = _run_broyden(problem, x0)
    _assert_broyden_result(
        solver, result, problem, x_star, rtol=5e-4, atol=5e-4
    )


def test_broyden_converges_strong_2d_nonsymmetric_nonlinear_residual() -> None:
    a = torch.tensor([[0.7, 0.25], [0.15, 0.9]], dtype=torch.float32)
    assert not torch.allclose(a, a.T)
    x_star = torch.tensor([[1.1], [0.6]], dtype=torch.float32)
    eps = 0.35
    x0 = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
    problem = NonlinearResidualProblem.from_known_root(a, x_star, eps)

    solver, result = _run_broyden(problem, x0, max_iterations=250)
    _assert_broyden_result(
        solver, result, problem, x_star, rtol=1e-3, atol=1e-3
    )


def test_broyden_history_required_for_triangular_cubic_residual() -> None:
    """For this triangular system, history is required for convergence.

    With ``A = [[3, 2], [0, 4]]`` and ``eps = 0.1``, the real root is unique,
    while the second Picard component has derivative ``-3 - 0.3 * x2**2``.
    """
    a = torch.tensor([[3.0, 2.0], [0.0, 4.0]], dtype=torch.float32)
    assert not torch.allclose(a, a.T)
    x_star = torch.tensor([[0.2], [-0.3]], dtype=torch.float32)
    eps = 0.1
    x0 = torch.zeros((2, 1), dtype=torch.float32)
    problem = CubicResidualProblem.from_known_root(a, x_star, eps)

    solver, result = _run_broyden(problem, x0, max_iterations=100)
    _assert_broyden_result(solver, result, problem, x_star)
