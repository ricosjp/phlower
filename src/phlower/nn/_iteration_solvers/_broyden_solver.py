from __future__ import annotations

import dataclasses as dc
from collections import deque
from functools import partial
from typing import NamedTuple

import torch
from phlower_tensor import PhlowerTensor
from phlower_tensor import functionals as functions
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn._interface_iteration_solver import (
    IFIterationSolver,
    IOptimizeProblem,
)
from phlower.nn._iteration_solvers._bicgstab_solver import (
    BiCGStabSolver,
    _bicgstab_core,
)
from phlower.settings._iteration_solver_setting import (
    BroydenSolverSetting,
    IPhlowerIterationSolverSetting,
)
from phlower.utils import get_logger

_logger = get_logger(__name__)


@dc.dataclass(frozen=True, init=False)
class _BroydenFixedParameters:
    # Relative threshold for rejecting unstable Broyden pairs.
    _EPS: float = 1e-8

    # Contract all elements to a scalar inner product.
    _INNER: str = "...,...->"

    # All update keys are packed into this single key for the adjoint solve,
    # so that the Krylov inner products cover the coupled system.
    _ADJOINT_KEY: str = "_broyden_adjoint"


def _inner(a: PhlowerTensor, b: PhlowerTensor) -> PhlowerTensor:
    return functions.einsum(
        _BroydenFixedParameters._INNER, a, b, dimension="auto"
    )


def _factors(
    history: deque[tuple[PhlowerTensor, PhlowerTensor]],
    eps: float = _BroydenFixedParameters._EPS,
) -> list[tuple[PhlowerTensor, PhlowerTensor, PhlowerTensor]]:
    """Rebuild H factors from retained (s, y) pairs in oldest-first order.

    An empty history gives H = I; pairs below the relative denominator
    threshold are skipped.
    """
    built: list[tuple[PhlowerTensor, PhlowerTensor, PhlowerTensor]] = []
    for s, y in history:
        Hy = y
        for s_i, left_i, denom_i in built:
            Hy = Hy + left_i * (_inner(s_i, Hy) / denom_i)
        denom = _inner(s, Hy)
        threshold = eps * torch.linalg.norm(s) * torch.linalg.norm(Hy)
        if torch.abs(denom) <= threshold:
            continue
        left = s - Hy
        built.append((s, left, denom))
    return built


def _apply_H(
    v: PhlowerTensor, history: deque[tuple[PhlowerTensor, PhlowerTensor]]
) -> PhlowerTensor:
    Hv = v
    for s, left, denom in _factors(history):
        Hv = Hv + left * (_inner(s, Hv) / denom)
    return Hv


def _insert_pair(
    s: PhlowerTensor,
    y: PhlowerTensor,
    history: deque[tuple[PhlowerTensor, PhlowerTensor]],
    eps: float = _BroydenFixedParameters._EPS,
) -> None:
    """Accept an (s, y) pair of iterate and residual differences when stable."""
    Hy = _apply_H(y, history)
    denom = _inner(s, Hy)
    threshold = eps * torch.linalg.norm(s) * torch.linalg.norm(Hy)
    if torch.abs(denom) <= threshold:
        return
    history.append((s, y))


def _apply_H_collection(
    R: IPhlowerTensorCollections,
    histories: dict[str, deque[tuple[PhlowerTensor, PhlowerTensor]]],
    keys: list[str],
) -> IPhlowerTensorCollections:
    """Apply each key's independent Broyden history to its residual."""
    return phlower_tensor_collection(
        {k: _apply_H(R[k], histories[k]) for k in keys}
    )


class _IterationState(NamedTuple):
    """Iteration status."""

    n_iterated: int
    is_converged: bool
    is_diverged: bool

    def is_finished(self, max_iterations: int) -> bool:
        if self.is_converged or self.is_diverged:
            return True
        if self.n_iterated >= max_iterations:
            return True
        return False

    def get_diverged_message(self, criteria: float) -> str:
        return (
            f"Broyden solver has diverged at iteration {self.n_iterated}."
            f" with criteria value {criteria}. "
        )


class _IterataionStateChecker:
    def __init__(
        self,
        max_iterations: int,
        convergence_threshold: float,
        divergence_threshold: float,
    ):
        self._max_iterations = max_iterations
        self._convergence_threshold = convergence_threshold
        self._divergence_threshold = divergence_threshold

    def examine(
        self,
        criteria: IPhlowerTensorCollections,
        current_state: _IterationState,
    ) -> _IterationState:

        n_iterated = current_state.n_iterated + 1
        is_diverged = criteria > self._divergence_threshold
        is_converged = criteria < self._convergence_threshold

        return _IterationState(
            n_iterated=n_iterated,
            is_converged=is_converged,
            is_diverged=is_diverged,
        )


def _broyden_core(
    problem: IOptimizeProblem,
    initial_values: IPhlowerTensorCollections,
    self: BroydenSolver,
) -> IPhlowerTensorCollections:
    h_inputs = initial_values
    v = h_inputs.mask(self._keys)
    self._check_finite(v, stage="initial values", iteration=0)

    histories = {k: deque(maxlen=self._memory_length) for k in self._keys}
    v_prev = None
    R_prev = None

    for _ in range(self.max_iterations):
        R = problem.gradient(
            h_inputs,
            update_keys=self._keys,
            operator_keys=self._operator_keys,
        )

        iteration = self._iterated_state.n_iterated + 1
        self._check_finite(R, stage="residual", iteration=iteration)
        criteria = R.apply(torch.linalg.norm)
        self._check_finite(criteria, stage="residual norm", iteration=iteration)

        if v_prev is not None:
            for k in self._keys:
                _insert_pair(
                    s=v[k] - v_prev[k],
                    y=R[k] - R_prev[k],
                    history=histories[k],
                )

        self._iterated_state = self._iteration_checker.examine(
            criteria=criteria, current_state=self._iterated_state
        )

        _finished = self._iterated_state.is_finished(self.max_iterations)

        v_next = v - _apply_H_collection(R, histories, self._keys)

        if _finished:
            break

        self._check_finite(v_next, stage="updated values", iteration=iteration)

        h_inputs = h_inputs | v_next
        v_prev, R_prev = v, R
        v = v_next

    if self._iterated_state.is_diverged:
        _diverged_msg = self._iterated_state.get_diverged_message(criteria)
        _logger.warning(_diverged_msg)

    if (
        self._iterated_state.is_diverged
        and self._exit_before_update_when_diverged
    ):
        _logger.info(
            "Exit before update."
            "Set exit_before_update_when_diverged as False "
            "to avoid this message."
        )
        return h_inputs.mask(self._keys)

    self._check_finite(
        v_next,
        stage="updated values",
        iteration=self._iterated_state.n_iterated,
    )
    h_inputs = h_inputs | v_next

    return h_inputs.mask(self._keys)


class _AdjointIdentity(torch.autograd.Function):
    """Identity in forward; the backward solves the adjoint system.

    Applied after the one-step re-evaluation z = x* - R(x*), the returned
    w with (dR/dx)^T w = grad_output then propagates through z to every
    tensor R depends on, including tensors held inside the problem.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        solve_adjoint: partial,
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ctx.solve_adjoint = solve_adjoint
        return tuple(t.view_as(t) for t in tensors)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        *grad_outputs: torch.Tensor,
    ) -> tuple:
        return (None, *ctx.solve_adjoint(grad_outputs))


def _solve_adjoint(
    problem: IOptimizeProblem,
    initial_values: IPhlowerTensorCollections,
    x_star: IPhlowerTensorCollections,
    self: BroydenSolver,
    grad_outputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Solve (dR/dx)^T w = grad_outputs at the converged point x*.

    The residual is re-evaluated here so that its graph is only kept
    while the backward runs.
    """
    keys = self._keys
    x_leaf = phlower_tensor_collection({k: x_star[k].detach() for k in keys})
    for k in keys:
        x_leaf[k].to_tensor().requires_grad_(True)
    with torch.enable_grad():
        R = problem.gradient(
            initial_values | x_leaf,
            update_keys=keys,
            operator_keys=self._operator_keys,
        )
    R_tensors = tuple(R[k].to_tensor() for k in keys)
    x_tensors = tuple(x_leaf[k].to_tensor() for k in keys)
    sizes = [t.numel() for t in x_tensors]

    def _unpack(packed: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(
            t.reshape(x.shape)
            for t, x in zip(packed.split(sizes), x_tensors, strict=True)
        )

    def _pack(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat([t.reshape(-1) for t in tensors])

    def rmatvec(
        v: IPhlowerTensorCollections,
        update_keys: list[str],
        operator_keys: list[str] | None = None,
    ) -> IPhlowerTensorCollections:
        grads = torch.autograd.grad(
            R_tensors,
            x_tensors,
            grad_outputs=_unpack(
                v[_BroydenFixedParameters._ADJOINT_KEY].to_tensor()
            ),
            retain_graph=True,
            allow_unused=True,
        )
        grads = tuple(
            torch.zeros_like(x) if g is None else g
            for g, x in zip(grads, x_tensors, strict=True)
        )
        return phlower_tensor_collection(
            {_BroydenFixedParameters._ADJOINT_KEY: _pack(grads)}
        )

    # Adjoint variables have units of dL/dx, so they are kept dimensionless.
    b = phlower_tensor_collection(
        {_BroydenFixedParameters._ADJOINT_KEY: _pack(grad_outputs)}
    )
    w0 = phlower_tensor_collection(
        {
            _BroydenFixedParameters._ADJOINT_KEY: torch.zeros_like(
                b[_BroydenFixedParameters._ADJOINT_KEY].to_tensor()
            )
        }
    )

    self._adjoint_solver.zero_residuals()
    (w,) = _bicgstab_core(rmatvec, b, w0, self._adjoint_solver)
    if not torch.isfinite(w).all():
        raise FloatingPointError(
            "Broyden solver encountered NaN or Inf in the adjoint solution. "
            "The Jacobian of the residual may be singular at the returned "
            "values."
        )
    return _unpack(w)


def _attach_implicit_gradient(
    problem: IOptimizeProblem,
    initial_values: IPhlowerTensorCollections,
    x_star: IPhlowerTensorCollections,
    self: BroydenSolver,
) -> IPhlowerTensorCollections:
    """Attach the implicit-function gradient to the converged values x*.

    Returns x* unchanged in value. Its gradient is
    dL/dtheta = w^T d(x - R)/dtheta with (dR/dx)^T w = dL/dx*.
    """
    keys = self._keys
    with torch.enable_grad():
        R = problem.gradient(
            initial_values | x_star,
            update_keys=keys,
            operator_keys=self._operator_keys,
        )
        z = phlower_tensor_collection({k: x_star[k] - R[k] for k in keys})

    if not any(z[k].to_tensor().requires_grad for k in keys):
        return x_star

    # Keep the converged values while gradients flow through z.
    out = phlower_tensor_collection(
        {k: x_star[k] + (z[k] - z[k].detach()) for k in keys}
    )
    solve_adjoint = partial(
        _solve_adjoint, problem, initial_values, x_star, self
    )
    tensors = _AdjointIdentity.apply(
        solve_adjoint, *[out[k].to_tensor() for k in keys]
    )
    return phlower_tensor_collection(
        {
            k: PhlowerTensor(
                t,
                dimension_tensor=out[k].dimension,
                is_time_series=out[k].is_time_series,
                is_voxel=out[k].is_voxel,
            )
            for k, t in zip(keys, tensors, strict=True)
        }
    )


class BroydenSolver(IFIterationSolver):
    @classmethod
    def from_setting(
        cls, setting: IPhlowerIterationSolverSetting
    ) -> BroydenSolver:
        assert isinstance(setting, BroydenSolverSetting)
        return BroydenSolver(**setting.model_dump())

    def __init__(
        self,
        max_iterations: int,
        convergence_threshold: float,
        divergence_threshold: float,
        update_keys: list[str],
        memory_length: int = 10,
        operator_keys: list[str] | None = None,
        exit_before_update_when_diverged: bool = False,
        exact_backward_flag: bool = True,
        adjoint_max_iterations: int = 100,
        adjoint_convergence_threshold: float = 1.0e-12,
        adjoint_divergence_threshold: float = 1.0e8,
    ) -> None:
        self._keys = update_keys

        self._memory_length = memory_length

        self._operator_keys = operator_keys
        self._exit_before_update_when_diverged = (
            exit_before_update_when_diverged
        )

        self._exact_backward_flag = exact_backward_flag

        self._iteration_checker = _IterataionStateChecker(
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            divergence_threshold=divergence_threshold,
        )

        self._iterated_state = _IterationState(
            n_iterated=0,
            is_converged=False,
            is_diverged=False,
        )

        # Thresholds apply to the squared residual norm of the adjoint system.
        if exact_backward_flag:
            self._adjoint_solver = None
        else:
            self._adjoint_solver = BiCGStabSolver(
                max_iterations=adjoint_max_iterations,
                convergence_threshold=adjoint_convergence_threshold,
                divergence_threshold=adjoint_divergence_threshold,
                update_keys=[_BroydenFixedParameters._ADJOINT_KEY],
                dict_variable_to_right={},
                dict_variable_to_dirichlet={},
            )

    @property
    def max_iterations(self) -> int:
        return self._iteration_checker._max_iterations

    @property
    def convergence_threshold(self) -> float:
        return self._iteration_checker._convergence_threshold

    @property
    def divergence_threshold(self) -> float:
        return self._iteration_checker._divergence_threshold

    def zero_residuals(self) -> None:
        self._iterated_state = _IterationState(
            n_iterated=0, is_converged=False, is_diverged=False
        )

    def get_n_iterated(self) -> int:
        return self._iterated_state.n_iterated

    def get_converged(self) -> bool:
        return self._iterated_state.is_converged

    def _check_finite(
        self,
        values: IPhlowerTensorCollections,
        *,
        stage: str,
        iteration: int,
    ) -> None:
        for key in values.keys():
            if not torch.isfinite(values[key].to_tensor()).all():
                self._iterated_state = _IterationState(
                    n_iterated=iteration,
                    is_converged=False,
                    is_diverged=True,
                )
                raise FloatingPointError(
                    f"Broyden solver encountered NaN or Inf in {stage} "
                    f"for key {key!r} at iteration {iteration}."
                )

    def run(
        self,
        initial_values: IPhlowerTensorCollections,
        problem: IOptimizeProblem,
    ) -> IPhlowerTensorCollections:
        """Run the iteration, raising FloatingPointError for nonfinite values.

        Checks cover the updated variables, residuals and their norms, and
        updates actually adopted, including the returned final update.

        With exact_backward_flag, gradients are backpropagated through all
        iterations. Otherwise, the iteration runs without a graph and the
        gradient at the returned values is given by the adjoint method.
        """
        if self._exact_backward_flag:
            return _broyden_core(problem, initial_values, self)

        with torch.no_grad():
            x_star = _broyden_core(problem, initial_values, self)
        x_star = x_star.apply(lambda v: v.detach())

        if not torch.is_grad_enabled():
            return x_star

        return _attach_implicit_gradient(problem, initial_values, x_star, self)
