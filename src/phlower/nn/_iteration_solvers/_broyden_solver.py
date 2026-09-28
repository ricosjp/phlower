from __future__ import annotations

from collections import deque
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
from phlower.settings._iteration_solver_setting import (
    BroydenSolverSetting,
    IPhlowerIterationSolverSetting,
)
from phlower.utils import get_logger

_logger = get_logger(__name__)

# Relative threshold for rejecting unstable Broyden pairs.
_EPS = 1e-8

# Contract all elements to a scalar inner product.
_INNER = "...,...->"


def _inner(a: PhlowerTensor, b: PhlowerTensor) -> PhlowerTensor:
    return functions.einsum(_INNER, a, b, dimension="auto")


def _factors(
    history: deque[tuple[PhlowerTensor, PhlowerTensor]], eps: float = _EPS
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
    eps: float = _EPS,
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
    ) -> None:
        self._keys = update_keys

        self._memory_length = memory_length

        self._operator_keys = operator_keys
        self._exit_before_update_when_diverged = (
            exit_before_update_when_diverged
        )

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
        """
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
            self._check_finite(
                criteria, stage="residual norm", iteration=iteration
            )

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

            self._check_finite(
                v_next, stage="updated values", iteration=iteration
            )

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
