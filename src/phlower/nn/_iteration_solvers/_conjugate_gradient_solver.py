from __future__ import annotations

from collections.abc import KeysView

import torch

from phlower.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.nn._core_modules import Dirichlet
from phlower.nn._functionals import _functions as functions
from phlower.nn._interface_iteration_solver import (
    IFIterationSolver,
    IOptimizeProblem,
)
from phlower.settings._iteration_solver_setting import (
    ConjugateGradientSolverSetting,
    IPhlowerIterationSolverSetting,
)
from phlower.utils import get_logger

_logger = get_logger(__name__)


class ConjugateGradientSolver(IFIterationSolver):
    @classmethod
    def from_setting(
        cls, setting: IPhlowerIterationSolverSetting
    ) -> ConjugateGradientSolver:
        assert isinstance(setting, ConjugateGradientSolverSetting)
        return ConjugateGradientSolver(**setting.__dict__)

    def __init__(
        self,
        max_iterations: int,
        convergence_threshold: float,
        divergence_threshold: float,
        update_keys: list[str],
        dict_variable_to_right: dict[str, str],
        dict_variable_to_dirichlet: dict[str, str],
        initial_convergence_threshold: float = 1.0e-16,
        operator_keys: list[str] | None = None,
    ) -> None:
        self._max_iterations = max_iterations
        self._convergence_threshold = convergence_threshold
        self._divergence_threshold = divergence_threshold
        self._update_keys = update_keys
        self._dict_variable_to_right = dict_variable_to_right
        self._dict_variable_to_dirichlet = dict_variable_to_dirichlet
        self._initial_convergence_threshold = initial_convergence_threshold
        self._operator_keys = operator_keys

        self._dict_dirichlet_module = {
            target: Dirichlet("identity", dirichlet_name=dirichlet)
            for target, dirichlet in self._dict_variable_to_dirichlet.items()
        }

        # internal status
        self._n_iterated = 0
        self._is_converged = False

        assert self._max_iterations > 1, "max_iterations should be > 1"
        self._validate_keys(
            "dict_variable_to_right", self._dict_variable_to_right.keys()
        )
        self._validate_keys(
            "dict_variable_to_dirichlet",
            self._dict_variable_to_dirichlet.keys(),
        )

    def _validate_keys(self, dict_name: str, dict_keys: KeysView):
        unmatched_keys = [
            dict_key
            for dict_key in dict_keys
            if dict_key not in self._update_keys
        ]
        if len(unmatched_keys) > 0:
            raise ValueError(
                f"Unmatched {dict_name}.\n"
                f"Given: {self._dict_variable_to_right.keys()}\n"
                f"Expected: {self._update_keys}"
            )

    def zero_residuals(self) -> None:
        self._n_iterated = 0
        self._is_converged = False

    def get_converged(self) -> bool:
        return self._is_converged

    def run(
        self,
        initial_values: IPhlowerTensorCollections,
        problem: IOptimizeProblem,
    ) -> IPhlowerTensorCollections:
        x = initial_values.clone()
        b = phlower_tensor_collection(
            {
                key: initial_values[self._dict_variable_to_right[key]]
                if key in self._dict_variable_to_right
                else torch.zeros_like(initial_values[key])
                for key in self._update_keys
            }
        )
        dirichlet = phlower_tensor_collection(
            {
                key: initial_values[dirichlet]
                for key, dirichlet in self._dict_variable_to_dirichlet.items()
            }
        )
        x = self._apply_dirichlet(x, dirichlet, factor=1.0)
        lap = problem.gradient(
            x, update_keys=self._update_keys, operator_keys=self._operator_keys
        )
        lap = self._apply_dirichlet(lap, dirichlet, factor=0.0)

        r = b - lap
        r = self._apply_dirichlet(r, dirichlet, factor=0.0)
        p = r
        norm_r = self._inner(r, r)

        initial_criteria = norm_r.apply(torch.linalg.norm)
        if initial_criteria < self._initial_convergence_threshold:
            return x.mask(self._update_keys)

        for _ in range(self._max_iterations):
            self._n_iterated += 1

            ap = problem.gradient(
                p,
                update_keys=self._update_keys,
                operator_keys=self._operator_keys,
            )
            ap = self._apply_dirichlet(ap, dirichlet, factor=0.0)
            alpha = norm_r / self._inner(p, ap)

            x.update(x.mask(self._update_keys) + alpha * p, overwrite=True)

            r = r - alpha * ap
            norm_r_new = self._inner(r, r)
            p = r + (norm_r_new / norm_r) * p
            norm_r = norm_r_new

            criteria = norm_r.apply(torch.linalg.norm) / initial_criteria
            if criteria < self._convergence_threshold:
                self._is_converged = True
                break

            if criteria > self._divergence_threshold:
                _logger.warning("CG solver has diverged.")
                self._is_converged = False
                break

        if not self._is_converged:
            _logger.warning("CG solver not converged.")
        return x.mask(self._update_keys)

    def _inner(
        self, a: IPhlowerTensorCollections, b: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        if len(a) != len(b):
            raise ValueError(
                "Length of a and b should be the same. "
                f"Given: {len(a)} and {len(b)}"
            )
        return phlower_tensor_collection(
            {k: functions.einsum("...,...->", a[k], b[k]) for k in a.keys()}
        )

    def _apply_dirichlet(
        self,
        u: IPhlowerTensorCollections,
        dirichlet: IPhlowerTensorCollections,
        factor: float = 1.0,
    ) -> IPhlowerTensorCollections:
        return phlower_tensor_collection(
            {
                k: self._dict_dirichlet_module[k](
                    phlower_tensor_collection(
                        {
                            k: u[k],
                            self._dict_variable_to_dirichlet[k]: dirichlet[k]
                            * factor,
                        }
                    )
                )
                for k in dirichlet.keys()
            }
        )
