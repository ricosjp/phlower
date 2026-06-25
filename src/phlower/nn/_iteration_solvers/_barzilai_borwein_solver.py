from __future__ import annotations

from collections import deque
from typing import Generic, Literal, NamedTuple, TypeVar

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
    BarzilaiBoweinSolverSetting,
    IPhlowerIterationSolverSetting,
)
from phlower.utils import get_logger

_logger = get_logger(__name__)

T = TypeVar("T")


class _FixedSizeMemory(Generic[T]):
    """This class is a wrapper of FIFO memory
     which has a fixed size.
    The oldest item is dropped to keep the size of memory
     is smaller than a fixed size when you add an item
    """

    def __init__(self, max_length: int):
        self._values = deque()
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._values)

    def append(self, value: T):
        self._values.append(value)
        if len(self._values) > self._max_length:
            self._values.popleft()

    @property
    def latest(self) -> T:
        return self._values[len(self) - 1]

    @property
    def oldest(self) -> T:
        return self._values[0]


class _IterationState(NamedTuple):
    """
    This class represents the state of iteration.
    """

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
            f"BB solver has diverged at iteration {self.n_iterated}."
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


# Barzilai–Borwein Method
class BarzilaiBorweinSolver(IFIterationSolver):
    @classmethod
    def from_setting(
        cls, setting: IPhlowerIterationSolverSetting
    ) -> BarzilaiBorweinSolver:
        assert isinstance(setting, BarzilaiBoweinSolverSetting)
        return BarzilaiBorweinSolver(**setting.model_dump())

    def __init__(
        self,
        max_iterations: int,
        convergence_threshold: float,
        divergence_threshold: float,
        update_keys: list[str],
        alpha_component_wise: bool = False,
        bb_type: Literal["long", "short"] = "long",
        operator_keys: list[str] | None = None,
        exit_before_update_when_diverged: bool = False,
        skip_last_update: bool = False,
    ) -> None:
        self._keys = update_keys

        self._alpha_calculator = AlphaCalculator(
            component_wise=alpha_component_wise,
            bb_type=bb_type,
        )

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

        # for backward compatibility
        self._skip_last_update = skip_last_update

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

    def run(
        self,
        initial_values: IPhlowerTensorCollections,
        problem: IOptimizeProblem,
    ) -> IPhlowerTensorCollections:
        v_history = _FixedSizeMemory(max_length=2)
        gradients_history = _FixedSizeMemory(max_length=2)

        assert self.max_iterations > 1
        h_inputs = initial_values
        v_next = h_inputs.mask(self._keys)

        for _ in range(self.max_iterations):
            gradient = problem.gradient(
                h_inputs,
                update_keys=self._keys,
                operator_keys=self._operator_keys,
            )

            v_history.append(v_next)
            gradients_history.append(gradient)

            criteria = gradient.apply(torch.linalg.norm)

            self._iterated_state = self._iteration_checker.examine(
                criteria=criteria, current_state=self._iterated_state
            )

            _finished = self._iterated_state.is_finished(self.max_iterations)

            v_next = self._compute_v_next(
                v_history=v_history,
                gradients_history=gradients_history,
            )

            if _finished:
                break

            # NOTE: update only considered variables
            h_inputs = h_inputs | v_next

        if self._iterated_state.is_diverged:
            _diverged_msg = self._iterated_state.get_diverged_message(criteria)
            _logger.warning(_diverged_msg)

        if self._skip_last_update:
            _logger.info(
                "Skip the last update for backward compatibility. "
                "Set skip_last_update as False to avoid this message."
            )
            return h_inputs.mask(self._keys)

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

        # NOTE: update only considered variables
        h_inputs = h_inputs | v_next
        return h_inputs.mask(self._keys)

    def _compute_v_next(
        self,
        v_history: _FixedSizeMemory,
        gradients_history: _FixedSizeMemory,
    ) -> IPhlowerTensorCollections:
        alphas = self._calculate_alphas(
            v_history.oldest,
            v_history.latest,
            gradients_history.oldest,
            gradients_history.latest,
        )

        # v_{i+1} = v_i - alpha_i * R(u_i)
        v_next = v_history.latest - alphas * gradients_history.latest
        return v_next

    def _calculate_alphas(
        self,
        v_prev: IPhlowerTensorCollections,
        v: IPhlowerTensorCollections,
        residual_prev: IPhlowerTensorCollections,
        residual: IPhlowerTensorCollections,
    ) -> IPhlowerTensorCollections:
        _dict_data = {
            k: self._alpha_calculator.calculate(
                delta_x=v[k] - v_prev[k], delta_g=residual[k] - residual_prev[k]
            )
            for k in self._keys
        }
        return phlower_tensor_collection(_dict_data)


class AlphaCalculator:
    def __init__(
        self,
        component_wise: bool,
        bb_type: Literal["long", "short"],
    ):
        self._componentwise_alpha = component_wise
        self._type_bb = bb_type

        self._zero_threshold = 1e-8

    def calculate(
        self, delta_x: PhlowerTensor, delta_g: PhlowerTensor
    ) -> PhlowerTensor | float:
        pattern = self._get_einsum_pattern()
        if self._type_bb == "long":
            if torch.any(
                torch.abs(
                    denominator := functions.einsum(pattern, delta_x, delta_g)
                )
                < self._zero_threshold
            ):
                return 1.0

            return functions.einsum(pattern, delta_x, delta_x) / denominator

        if self._type_bb == "short":
            if torch.any(
                (denominator := functions.einsum(pattern, delta_g, delta_g))
                < self._zero_threshold
            ):
                return 1.0

            return functions.einsum(pattern, delta_x, delta_g) / denominator

        raise NotImplementedError(
            f"bb type: {self._type_bb} is not implemented."
        )

    def _get_einsum_pattern(self) -> str:
        if self._componentwise_alpha:
            return "n...f,n...f->f"
        else:
            return "...,...->"
