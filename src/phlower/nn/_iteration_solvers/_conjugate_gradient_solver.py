from __future__ import annotations

from collections.abc import KeysView

import torch
from phlower_tensor import functionals as functions
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn._core_modules import Dirichlet
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


def _cg_core(
    problem: IOptimizeProblem,
    b: IPhlowerTensorCollections,
    initial_values: IPhlowerTensorCollections,
    self: ConjugateGradientSolver,
) -> tuple[torch.Tensor]:
    x = initial_values.clone()
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
    if initial_criteria >= self._initial_convergence_threshold:
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

    outputs = []
    for key in initial_values.keys():
        if key in x.keys():
            outputs.append(x[key]._tensor)
        else:
            outputs.append(initial_values[key]._tensor)

    return_vals = tuple(outputs)

    return return_vals


class ConjugateGradientSolver_Core(torch.autograd.Function):
    clear_saved_tensors_on_access = True

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, *args: tuple
    ) -> tuple[torch.Tensor]:
        num_tensors = len(args) - 4
        tensors = args[:num_tensors]
        initial_values = args[num_tensors]
        problem = args[num_tensors + 1]
        self = args[num_tensors + 2]

        keys = initial_values.keys()
        assert num_tensors == len(keys)

        initial_values = phlower_tensor_collection(
            dict(zip(keys, tensors, strict=True))
        )

        b = phlower_tensor_collection(
            {
                key: initial_values[self._dict_variable_to_right[key]]
                if key in self._dict_variable_to_right
                else torch.zeros_like(initial_values[key])
                for key in self._update_keys
            }
        )

        ctx.phlower_iteration_problem = problem
        ctx.phlower_gradient_solver = self
        ctx.phlower_iteration_problem_initial_values = initial_values

        x = _cg_core(problem, b, initial_values, self)

        x_clone = tuple([t.detach().clone().requires_grad_(False) for t in x])

        ctx.save_for_backward(*x_clone)

        return x

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        *grad_outputs_core: tuple[torch.Tensor],
    ) -> tuple:
        problem = ctx.phlower_iteration_problem
        self = ctx.phlower_gradient_solver
        keys = ctx.phlower_iteration_problem_initial_values.keys()
        x_tensors = ctx.saved_tensors

        initial_values_zero = phlower_tensor_collection(
            {
                key: ctx.phlower_iteration_problem_initial_values[key] * 0.0
                for key in keys
            }
        )

        grad_outputs = phlower_tensor_collection(
            dict(zip(keys, grad_outputs_core, strict=True))
        )
        grad_outputs = phlower_tensor_collection(
            {
                key: torch.zeros_like(initial_values_zero[key])
                if key not in self._dict_variable_to_right
                else grad_outputs[key]
                for key in self._update_keys
            }
        )

        grad_b = _cg_core(
            problem, grad_outputs, initial_values_zero, self
        )  # because A = A^T

        grad_b_new = dict(zip(keys, grad_b, strict=True))
        for key, right_key in self._dict_variable_to_right.items():
            grad_b_new[right_key] = grad_b_new[right_key] + grad_b_new[key]

        grad_b = tuple(grad_b_new.values())

        grad_b_core = tuple(
            [
                grad_b_tmp
                for key, grad_b_tmp in zip(keys, grad_b, strict=True)
                if key in self._update_keys
            ]
        )

        new_x_tensors = x_tensors

        with torch.enable_grad():
            new_x = phlower_tensor_collection(
                dict(zip(keys, new_x_tensors, strict=True))
            )
            c = problem.gradient(
                new_x,
                update_keys=self._update_keys,
                operator_keys=self._operator_keys,
            )  # The operator is not executing the processing in _cg_core
            tensors = []
            grad_tensors = []
            c_requires_grad = False
            for key, grad_b_tmp in zip(
                self._update_keys, grad_b_core, strict=True
            ):
                c_requires_grad = c_requires_grad | c[key]._tensor.requires_grad
                tensors.append(c[key]._tensor)
                grad_tensors.append(-grad_b_tmp)
            if c_requires_grad:
                torch.autograd.backward(tensors, grad_tensors=grad_tensors)

        return grad_b + tuple(None for _ in range(4))


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
        exact_backward_flag: bool = True,
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
        self._exact_backward_flag = exact_backward_flag

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
        keys = initial_values.keys()

        if self._exact_backward_flag:
            b = phlower_tensor_collection(
                {
                    key: initial_values[self._dict_variable_to_right[key]]
                    if key in self._dict_variable_to_right
                    else torch.zeros_like(initial_values[key])
                    for key in self._update_keys
                }
            )
            outputs = _cg_core(problem, b, initial_values, self)
        else:
            tensors = [initial_values[key]._tensor for key in keys]
            dummy_tensor = torch.tensor(0.0, requires_grad=True)
            combined_args = (
                tuple(tensors)
                + (initial_values,)
                + (problem,)
                + (self,)
                + (dummy_tensor,)
            )
            outputs = ConjugateGradientSolver_Core.apply(*combined_args)

        for output, key in zip(outputs, keys, strict=True):
            x.update({key: output}, overwrite=True)

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
