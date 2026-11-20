"""Tests for the adjoint (implicit) gradient of the Broyden solver."""

from __future__ import annotations

import gc

import pytest
import torch
from phlower_tensor import phlower_tensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn._group_module import _GroupOptimizeProblem
from phlower.nn._interface_iteration_solver import IOptimizeProblem
from phlower.nn._iteration_solvers._broyden_solver import BroydenSolver
from phlower.settings._iteration_solver_setting import BroydenSolverSetting

_DT = 0.1
_N = 3
_KEYS = ["u", "v"]
_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _make_parameters(requires_grad: bool = True) -> dict[str, torch.Tensor]:

    generator = torch.Generator(device=_DEVICE).manual_seed(0)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, generator=generator, dtype=torch.float64, device=_DEVICE
        ).requires_grad_(requires_grad)

    return {
        "A": rand(_N, _N),
        "B": rand(_N, _N),
        "c": rand(1),
        "f": rand(_N, 1),
        "u0": rand(_N, 1),
        "v0": rand(_N, 1),
    }


def _make_initials(
    params: dict[str, torch.Tensor],
) -> IPhlowerTensorCollections:
    # "f" is an input that is not updated by the solver.
    return phlower_tensor_collection(
        {
            "u": phlower_tensor(params["u0"]),
            "v": phlower_tensor(params["v0"]),
            "f": phlower_tensor(params["f"]),
        }
    )


def _make_problem(
    params: dict[str, torch.Tensor], initials: IPhlowerTensorCollections
) -> _GroupOptimizeProblem:
    # Implicit Euler step of a coupled nonlinear system.
    # "A", "B" and "c" are only visible inside the problem.
    def step_forward(
        h: IPhlowerTensorCollections,
    ) -> IPhlowerTensorCollections:
        u = h["u"].to_tensor()
        v = h["v"].to_tensor()
        f = h["f"].to_tensor()
        du = torch.tanh(params["A"] @ u + params["c"] * v) + f
        dv = torch.sin(params["B"] @ v) - 0.5 * u * u
        return phlower_tensor_collection(
            {
                "u": phlower_tensor(u + _DT * du),
                "v": phlower_tensor(v + _DT * dv),
            }
        )

    return _GroupOptimizeProblem(
        initials=initials, step_forward=step_forward, steady_mode=False
    )


def _make_solver(exact_backward_flag: bool) -> BroydenSolver:
    return BroydenSolver(
        max_iterations=100,
        convergence_threshold=1.0e-13,
        divergence_threshold=1.0e8,
        update_keys=_KEYS,
        exact_backward_flag=exact_backward_flag,
        adjoint_convergence_threshold=1.0e-24,
    )


def _loss(result: IPhlowerTensorCollections) -> torch.Tensor:
    u = result["u"].to_tensor()
    v = result["v"].to_tensor()
    weight = torch.arange(1, _N + 1, dtype=u.dtype, device=_DEVICE).reshape(
        _N, 1
    )
    return (u * weight).sum() + (v * v).sum()


def _solve_and_differentiate(
    exact_backward_flag: bool,
) -> tuple[IPhlowerTensorCollections, dict[str, torch.Tensor]]:
    params = _make_parameters()
    initials = _make_initials(params)
    problem = _make_problem(params, initials)
    solver = _make_solver(exact_backward_flag)

    result = solver.run(initials, problem)
    assert solver.get_converged()

    grads = torch.autograd.grad(_loss(result), list(params.values()))
    return result, dict(zip(params.keys(), grads, strict=True))


def _reference_gradients() -> dict[str, torch.Tensor]:
    """Implicit gradient from the dense Jacobian at the converged point."""
    params = _make_parameters()
    initials = _make_initials(params)
    problem = _make_problem(params, initials)
    with torch.no_grad():
        result = _make_solver(exact_backward_flag=True).run(initials, problem)

    x_star = {k: result[k].to_tensor().detach() for k in _KEYS}
    sizes = [x_star[k].numel() for k in _KEYS]

    def residual(packed: torch.Tensor) -> torch.Tensor:
        x = {
            k: t.reshape(x_star[k].shape)
            for k, t in zip(_KEYS, packed.split(sizes), strict=True)
        }
        R = problem.gradient(
            initials | phlower_tensor_collection(x), update_keys=_KEYS
        )
        return torch.cat([R[k].to_tensor().reshape(-1) for k in _KEYS])

    packed = torch.cat([x_star[k].reshape(-1) for k in _KEYS])
    jacobian = torch.autograd.functional.jacobian(residual, packed)

    packed_leaf = packed.clone().requires_grad_(True)
    x_leaf = {
        k: t.reshape(x_star[k].shape)
        for k, t in zip(_KEYS, packed_leaf.split(sizes), strict=True)
    }
    (dloss_dx,) = torch.autograd.grad(
        _loss(phlower_tensor_collection(x_leaf)), packed_leaf
    )
    adjoint = torch.linalg.solve(jacobian.T, dloss_dx)

    grads = torch.autograd.grad(
        residual(packed), list(params.values()), grad_outputs=-adjoint
    )
    return dict(zip(params.keys(), grads, strict=True))


def test__adjoint_gradient_matches_unrolled_gradient():
    _, unrolled = _solve_and_differentiate(exact_backward_flag=True)
    _, adjoint = _solve_and_differentiate(exact_backward_flag=False)

    for key in unrolled:
        torch.testing.assert_close(
            adjoint[key], unrolled[key], rtol=1e-6, atol=1e-9, msg=key
        )


@pytest.mark.gpu_test
def test__check_memory_usage():
    _clear_cache()
    _, _ = _solve_and_differentiate(exact_backward_flag=False)
    low_memory_max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    del _

    _clear_cache()
    _, _ = _solve_and_differentiate(exact_backward_flag=True)
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    del _

    assert (
        low_memory_max_allocated * 1.05 < max_allocated
        or not torch.cuda.is_available()
    )


def test__adjoint_gradient_matches_dense_implicit_gradient():
    _, adjoint = _solve_and_differentiate(exact_backward_flag=False)
    reference = _reference_gradients()

    for key in reference:
        assert adjoint[key] is not None
        assert reference[key] is not None

        torch.testing.assert_close(
            adjoint[key], reference[key], rtol=1e-8, atol=1e-10, msg=key
        )


def test__adjoint_keeps_forward_values():
    unrolled, _ = _solve_and_differentiate(exact_backward_flag=True)
    adjoint, _ = _solve_and_differentiate(exact_backward_flag=False)

    for key in _KEYS:
        torch.testing.assert_close(
            adjoint[key].to_tensor(), unrolled[key].to_tensor()
        )


def test__adjoint_output_does_not_require_grad_without_parameters():
    params = _make_parameters(requires_grad=False)
    initials = _make_initials(params)
    problem = _make_problem(params, initials)

    result = _make_solver(exact_backward_flag=False).run(initials, problem)

    assert all(not result[k].to_tensor().requires_grad for k in _KEYS)


def test__adjoint_output_does_not_require_grad_under_no_grad():
    params = _make_parameters()
    initials = _make_initials(params)
    problem = _make_problem(params, initials)

    with torch.no_grad():
        result = _make_solver(exact_backward_flag=False).run(initials, problem)

    assert all(not result[k].to_tensor().requires_grad for k in _KEYS)


class _ScalarAffineProblem(IOptimizeProblem):
    """``R(x, theta) = a*x - theta`` with a single element."""

    def __init__(self, a: float, theta: torch.Tensor) -> None:
        self._a = a
        self._theta = theta

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
        x = h["x"].to_tensor()
        return phlower_tensor_collection(
            {"x": phlower_tensor(self._a * x - self._theta)}
        )


def _run_scalar_affine(a: float) -> tuple[torch.Tensor, torch.Tensor]:
    theta = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    problem = _ScalarAffineProblem(a=a, theta=theta)
    solver = BroydenSolver(
        max_iterations=20,
        convergence_threshold=1.0e-12,
        divergence_threshold=1.0e8,
        update_keys=["x"],
        exact_backward_flag=False,
    )
    initial = phlower_tensor_collection(
        {"x": phlower_tensor(torch.zeros(1, dtype=torch.float64))}
    )
    result = solver.run(initial, problem)["x"].to_tensor()
    assert solver.get_converged()
    return result, theta


@pytest.mark.parametrize("a", [3.0, 0.5])
def test__adjoint_gradient_of_scalar_problem(a: float):
    # The adjoint system is scalar, which used to break BiCGStab down.
    result, theta = _run_scalar_affine(a)

    (gradient,) = torch.autograd.grad(result.sum(), theta)

    torch.testing.assert_close(
        gradient, torch.tensor([1.0 / a], dtype=torch.float64)
    )


@pytest.mark.parametrize("exact_backward_flag", [True, False])
def test__exact_backward_flag_from_setting(exact_backward_flag: bool):
    setting = BroydenSolverSetting(
        update_keys=_KEYS, exact_backward_flag=exact_backward_flag
    )

    solver = BroydenSolver.from_setting(setting)

    assert solver._exact_backward_flag is exact_backward_flag
