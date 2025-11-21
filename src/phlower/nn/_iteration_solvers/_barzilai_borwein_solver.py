from __future__ import annotations

from collections import deque
from typing import Generic, Literal, TypeVar

import torch

from phlower._base.tensors import PhlowerTensor
from phlower.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.nn._functionals import _functions as functions
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
    ) -> None:
        self._max_iterations = max_iterations
        self._convergence_threshold = convergence_threshold
        self._divergence_threshold = divergence_threshold
        self._keys = update_keys

        self._alpha_calculator = AlphaCalculator(
            component_wise=alpha_component_wise,
            bb_type=bb_type,
        )

        self._operator_keys = operator_keys
        # internal status
        self._n_iterated = 0
        self._is_converged = False

    def zero_residuals(self) -> None:
        self._n_iterated = 0
        self._is_converged = False

    def get_n_iterated(self) -> int:
        return self._n_iterated

    def get_converged(self) -> bool:
        return self._is_converged

    def run(
        self,
        initial_values: IPhlowerTensorCollections,
        problem: IOptimizeProblem,
    ) -> IPhlowerTensorCollections:
        v_history = _FixedSizeMemory(max_length=2)
        gradients_history = _FixedSizeMemory(max_length=2)

        assert self._max_iterations > 1
        h_inputs = initial_values
        v_next = h_inputs.mask(self._keys)

        for idx in range(self._max_iterations):
            self._n_iterated += 1

            if idx != 0:
                alphas = self._calculate_alphas(
                    v_history.oldest,
                    v_history.latest,
                    gradients_history.oldest,
                    gradients_history.latest,
                )

                # v_{i+1} = v_i - alpha_i * R(u_i)
                v_next = v_history.latest - alphas * gradients_history.latest

                # NOTE: update only considered variables
                h_inputs.update(v_next, overwrite=True)

            gradient = problem.gradient(
                h_inputs,
                update_keys=self._keys,
                operator_keys=self._operator_keys,
            )

            v_history.append(v_next)
            gradients_history.append(gradient)

            criteria = gradient.apply(torch.linalg.norm)
            if criteria < self._convergence_threshold:
                self._is_converged = True
                break

            if criteria > self._divergence_threshold:
                _logger.warning("BB solver has diverged.")
                self._is_converged = False
                break

        return h_inputs.mask(self._keys)

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
