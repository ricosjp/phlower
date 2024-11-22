from collections import deque
from types import TracebackType
from typing import Generic, Literal, TypeVar

import torch
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower._fields._simulation_field import ISimulationField
from phlower.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.nn._core_modules import _functions as functions
from phlower.nn._interface_module import IPhlowerGroup
from phlower.nn._iteration_solver import IFIterationSolver

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


class ImplicitIterationSolver(IFIterationSolver):
    def __init__(self) -> None:
        self._max_iterations = 10
        self._criteria: float = 0.0
        self._keys = []
        self.steady = False

        self._alpha_calculator = AlphaCalculator(learn_alpha=False)

    def __enter__(self) -> Self:
        self._residuals = deque()
        return self

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ):
        return None

    def get_converged(self) -> bool: ...

    def run(
        self,
        group: IPhlowerGroup,
        *,
        inputs: IPhlowerTensorCollections,
        field_data: ISimulationField,
        **kwards,
    ) -> IPhlowerTensorCollections:
        v_history = _FixedSizeMemory(max_length=2)
        residuals_history = _FixedSizeMemory(max_length=2)

        assert self._max_iterations > 1
        h_inputs = inputs

        for idx in self._max_iterations:
            if idx != 0:
                alphas = self._calculate_alphas(
                    v_history.oldest,
                    v_history.latest,
                    residuals_history.oldest,
                    residuals_history.latest,
                )

                # v_{i+1} = v_i - alpha_i * R(u_i)
                v_next = v_history.latest - alphas * residuals_history.latest

                # NOTE: update only considered variables
                h_inputs.update(v_next)

                if self.get_converged():
                    break

            operator_value = group.step_forward(
                h_inputs, field_data=field_data, **kwards
            )
            residuals = self._calculate_residuals(
                h_inputs.mask(self._keys),
                inputs.mask(self._keys),
                operator_value.mask(self._keys),
            )

            v_history.append(v_next)
            residuals_history.append(residuals)

        return h_inputs

    def _calculate_residuals(
        self,
        h: IPhlowerTensorCollections,
        inputs: IPhlowerTensorCollections,
        operator_value: IPhlowerTensorCollections,
    ) -> IPhlowerTensorCollections:
        if self.steady:
            # R(u) = - D[u] dt
            residuals = -1.0 * operator_value
        else:
            # R(u) = v - u(t) - D[u] dt
            residuals = h - inputs - operator_value

        return residuals

    def _calculate_alphas(
        self,
        v_prev: IPhlowerTensorCollections,
        v: IPhlowerTensorCollections,
        residual_prev: IPhlowerTensorCollections,
        residual: IPhlowerTensorCollections,
    ) -> IPhlowerTensorCollections:
        _dict_data = {
            self._alpha_calculator.calculate(
                delta_x=v[k] - v_prev[k], delta_g=residual[k] - residual_prev[k]
            )
            for k in self._keys
        }
        return phlower_tensor_collection(_dict_data)


class AlphaCalculator:
    def __init__(self, learn_alpha: bool):
        self._learn_alpha = learn_alpha
        self._alpha_model: torch.Module | None = None
        self._componentwise_alpha: bool = False
        self._type_bb: Literal["long", "short"] = "long"

        self._zero_threshold = 1e-8

    def calculate(
        self, delta_x: PhlowerTensor, delta_g: PhlowerTensor
    ) -> PhlowerTensor | float:
        if self._learn_alpha:
            return self._alpha_model(
                torch.cat(
                    [
                        functions.einsum("n...f,n...f->f", delta_x, delta_x),
                        functions.einsum("n...f,n...f->f", delta_g, delta_g),
                        functions.einsum("n...f,n...f->f", delta_x, delta_g),
                    ],
                    dim=-1,
                )
            )

        if (
            torch.sum(delta_x) < self._zero_threshold
            and torch.sum(delta_g) < self._zero_threshold
        ):
            return 1.0

        pattern = self._get_einsum_pattern()
        if self._type_bb == "long":
            return functions.einsum(pattern, delta_x, delta_x) / (
                functions.einsum(pattern, delta_x, delta_g) + 1.0e-5
            )
        if self._type_bb == "short":
            return functions.einsum(pattern, delta_x, delta_g) / (
                functions.einsum(pattern, delta_g, delta_g) + 1.0e-5
            )

        raise NotImplementedError(
            f"bb type: {self._type_bb} is not implemented."
        )

    def _get_einsum_pattern(self) -> str:
        if self._componentwise_alpha:
            return "n...f,n...f->f"
        else:
            return "...,...->"
