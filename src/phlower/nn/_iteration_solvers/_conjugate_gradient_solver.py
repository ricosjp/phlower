from __future__ import annotations

import abc
from collections.abc import KeysView

import pydantic
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
    CGPreconditionParameters,
    ConjugateGradientSolverSetting,
    IPhlowerIterationSolverSetting,
)
from phlower.utils import get_logger
from phlower.utils.enums import PhlowerCGPreconditionType

_logger = get_logger(__name__)


# region Preconditioner


def create_preconditioner(
    precondition_type: None | str,
    parameteres: dict | pydantic.BaseModel | None = None,
) -> IPreconditioner:
    precondtioner_type = precondition_type or PhlowerCGPreconditionType.none
    parameters = parameteres or {}
    if isinstance(parameters, pydantic.BaseModel):
        parameters = parameters.model_dump()

    match precondtioner_type:
        case PhlowerCGPreconditionType.none:
            return NonePreconditioner(**parameters)

        case PhlowerCGPreconditionType.random_jacobi:
            return RandomJacobiPreconditioner(**parameters)

        case _:
            raise NotImplementedError(
                f"Preconditioner type {precondtioner_type} is not implemented."
            )


class IPreconditioner(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build_internal_state(
        self,
        solver: ConjugateGradientSolver,
        problem: IOptimizeProblem,
        initial_values: IPhlowerTensorCollections,
    ) -> None:
        """
        Prepare the preconditioner for the iteration solver.
        This method should be called before the iteration solver starts.
        In many cases, precondition matrix is constructed
        based on the initial values and the problem.
        """
        ...

    @abc.abstractmethod
    def apply(
        self, x: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections: ...


class NonePreconditioner(IPreconditioner):
    def __init__(self):
        return

    def build_internal_state(
        self,
        solver: ConjugateGradientSolver,
        problem: IOptimizeProblem,
        initial_values: IPhlowerTensorCollections,
    ) -> None:
        return

    def apply(self, x: IPhlowerTensorCollections) -> IPhlowerTensorCollections:
        return x


# TODO
# Preconditioner should preferably not depend on
# the specific linear solver being used.
class RandomJacobiPreconditioner(IPreconditioner):
    """
    RandomJacobiPreconditioner is a Jacobi preconditioner
    that estimates the diagonal of a LinearOperator via a randomized algorithm.

    Ref: https://arxiv.org/pdf/2202.02887
    """

    def __init__(
        self,
        num_of_trials: int = 1,
    ) -> None:
        self.num_of_trials = num_of_trials
        self.pred_diag: IPhlowerTensorCollections | None = None

    def build_internal_state(
        self,
        solver: ConjugateGradientSolver,
        problem: IOptimizeProblem,
        initial_values: IPhlowerTensorCollections,
    ) -> None:
        # TODO: this process heavily depends on CG solver's implementation.
        # We need to find a better way to implement this.

        key_shape_pair = {
            key: initial_values[key].shape for key in solver._update_keys
        }
        dirichlet = phlower_tensor_collection(
            {
                key: initial_values[dirichlet]
                for key, dirichlet in solver._dict_variable_to_dirichlet.items()
            }
        )

        pred2 = None
        for _ in range(self.num_of_trials):
            y = phlower_tensor_collection(
                {
                    key: torch.randn(shape, requires_grad=False)
                    for key, shape in key_shape_pair.items()
                }
            )

            y = solver._apply_dirichlet(y, dirichlet, factor=0.0)

            lap = (
                problem.gradient(
                    y,
                    update_keys=solver._update_keys,
                    operator_keys=solver._operator_keys,
                )
                * y
            )
            lap = solver._apply_dirichlet(lap, dirichlet, factor=0.0)
            if pred2 is None:
                pred2 = lap
            else:
                pred2 = pred2 + lap

        self.pred_diag = pred2 / self.num_of_trials

    def apply(self, x: IPhlowerTensorCollections) -> IPhlowerTensorCollections:
        if self.pred_diag is None:
            raise ValueError(
                "Preconditioner is not initialized. "
                "Call `build_internal_state` method "
                "before applying the preconditioner."
            )

        ans = {}
        assert self.pred_diag.keys() == x.keys()
        eps = 1.0e-6
        for key in x.keys():
            ans[key] = torch.where(
                torch.abs(self.pred_diag[key]._tensor) > eps,
                x[key] / self.pred_diag[key],
                x[key],
            )

        return phlower_tensor_collection(ans)


# endregion


def _cg_core(
    problem: IOptimizeProblem,
    b: IPhlowerTensorCollections,
    initial_values: IPhlowerTensorCollections,
    self: ConjugateGradientSolver,
    preconditioner: IPreconditioner | None = None,
) -> tuple[torch.Tensor]:
    if preconditioner is None:
        preconditioner = NonePreconditioner()

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
    z = x | preconditioner.apply(r)
    z = self._apply_dirichlet(z, dirichlet, factor=0.0)
    p = z
    norm_rz = self._inner(r.mask(self._update_keys), z.mask(self._update_keys))

    initial_criteria = norm_rz.apply(torch.linalg.norm)
    if initial_criteria >= self._initial_convergence_threshold:
        for _ in range(self._max_iterations):
            self._n_iterated += 1

            ap = problem.gradient(
                p,
                update_keys=self._update_keys,
                operator_keys=self._operator_keys,
            )
            ap = self._apply_dirichlet(ap, dirichlet, factor=0.0)
            alpha = norm_rz / self._inner(p.mask(self._update_keys), ap)

            x.update(
                x.mask(self._update_keys) + alpha * p.mask(self._update_keys),
                overwrite=True,
            )

            r = r - alpha * ap
            z = preconditioner.apply(r)
            norm_rz_new = self._inner(r, z)
            p.update(
                r + (norm_rz_new / norm_rz) * p.mask(self._update_keys),
                overwrite=True,
            )
            norm_rz = norm_rz_new

            criteria = norm_rz.apply(torch.linalg.norm) / initial_criteria

            if criteria < self._convergence_threshold:
                self._is_converged = True
                break

            if criteria > self._divergence_threshold:
                _logger.warning("CG solver has diverged.")
                self._is_converged = False
                break

        if not self._is_converged:
            total_residual = sum(criteria.values()).detach().to("cpu").numpy()
            message = (
                "CG solver not converged. "
                f"iter: {self._n_iterated}, residual: {total_residual}"
            )
            if self._log_level == "warning":
                _logger.warning(message)
            elif self._log_level == "info":
                _logger.info(message)
            elif self._log_level == "debug":
                _logger.debug(message)
            else:
                raise ValueError(f"Invalid log level: {self._log_level}")

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

        x = _cg_core(
            problem,
            b,
            initial_values,
            self,
            preconditioner=self._preconditioner,
        )

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
            problem,
            grad_outputs,
            initial_values_zero,
            self,
            preconditioner=self._preconditioner,
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
        return ConjugateGradientSolver(**setting.model_dump())

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
        log_level: str = "warning",
        precondition_type: str | None = None,
        precondition_parameters: dict | CGPreconditionParameters | None = None,
    ) -> None:
        self._max_iterations = max_iterations
        self._convergence_threshold = convergence_threshold
        self._divergence_threshold = divergence_threshold
        self._update_keys = update_keys
        self._dict_variable_to_right = dict_variable_to_right
        self._dict_variable_to_dirichlet = dict_variable_to_dirichlet
        self._initial_convergence_threshold = initial_convergence_threshold
        self._operator_keys = operator_keys
        self._log_level = log_level

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

        self._precondition_type = precondition_type
        self._precondition_parameters = precondition_parameters or {}
        self._preconditioner = create_preconditioner(
            precondition_type=precondition_type,
            parameteres=self._precondition_parameters,
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
        keys = initial_values.keys()

        with torch.no_grad():
            self._preconditioner.build_internal_state(
                solver=self,
                problem=problem,
                initial_values=initial_values,
            )
            # The same preconditioner is applied to both forward and backward
            # passes because A=A^T

        if self._exact_backward_flag:
            b = phlower_tensor_collection(
                {
                    key: initial_values[self._dict_variable_to_right[key]]
                    if key in self._dict_variable_to_right
                    else torch.zeros_like(initial_values[key])
                    for key in self._update_keys
                }
            )

            outputs = _cg_core(
                problem,
                b,
                initial_values,
                self,
                preconditioner=self._preconditioner,
            )
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
        updated_data = phlower_tensor_collection(
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
        return u | updated_data
