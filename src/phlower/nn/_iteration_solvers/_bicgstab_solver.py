from __future__ import annotations

from collections.abc import Callable, KeysView
from functools import partial

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
from phlower.nn._iteration_solvers._conjugate_gradient_solver import (
    CGPreconditionParameters,
    IPreconditioner,
    NonePreconditioner,
    create_preconditioner,
)
from phlower.settings._iteration_solver_setting import (
    BiCGStabSolverSetting,
    IPhlowerIterationSolverSetting,
)
from phlower.utils import get_logger

_logger = get_logger(__name__)


def rmatvec_core(
    problem: IOptimizeProblem,
    v: IPhlowerTensorCollections,
    update_keys: list[str],
    operator_keys: list[str] | None = None,
) -> IPhlowerTensorCollections:
    """
    Compute the product of the transpose of the Linear operator and a vector v
    without explicitly forming the matrix.

    ``problem.gradient`` is used to compute the product of the Linear operator
    and a vector v.

    Args:
        problem (IOptimizeProblem): The optimization problem that defines the
            linear operator.
        v (IPhlowerTensorCollections): The vector to be multiplied by the
            transpose of the linear operator.
        update_keys (list[str]): The keys of the variables to be updated.
        operator_keys (list[str] | None): The keys of the variables that are
            used in the operator. If None, all variables are used.

    Returns:
        IPhlowerTensorCollections: The result of the multiplication of the
            transpose of the linear operator and the vector v.
    """
    with torch.enable_grad():
        dummy_x = phlower_tensor_collection(
            {
                key: torch.zeros_like(value, requires_grad=True)
                for key, value in v.items()
            }
        )
        y = problem.gradient(
            dummy_x,
            update_keys=update_keys,
            operator_keys=operator_keys,
        )
        dummy_x_list = [dummy_x[key].to_tensor() for key in dummy_x.keys()]

        v_list = [v[key].to_tensor() for key in y.keys()]
        y_list = [y[key].to_tensor() for key in y.keys()]

        # Compute the product of the transpose of the linear operator
        # and the vector v via the chain rule of differentiation.
        # A x = y, then A^T v = (dy/dx)^T v
        ans_list = torch.autograd.grad(
            outputs=y_list,
            inputs=dummy_x_list,
            grad_outputs=v_list,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        ans = phlower_tensor_collection(
            {
                key: tensor
                for key, tensor in zip(v.keys(), ans_list, strict=True)
                if tensor is not None
            }
        )
        ans = ans.mask(update_keys)
    return ans


def _bicgstab_core(
    gradient: Callable[
        [IPhlowerTensorCollections, list[str], list[str] | None],
        IPhlowerTensorCollections,
    ],
    b: IPhlowerTensorCollections,
    initial_values: IPhlowerTensorCollections,
    self: BiCGStabSolver,
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
    lap = gradient(
        x, update_keys=self._update_keys, operator_keys=self._operator_keys
    )
    lap = self._apply_dirichlet(lap, dirichlet, factor=0.0)

    # r = b-Ax
    r = b - lap
    r = x | self._apply_dirichlet(r, dirichlet, factor=0.0)

    norm_r = self._inner(r.mask(self._update_keys), r.mask(self._update_keys))
    initial_criteria = norm_r.apply(torch.linalg.norm)

    # r0=r
    r0 = phlower_tensor_collection(
        {key: tensor.detach().clone() for key, tensor in r.items()}
    )

    if initial_criteria >= self._initial_convergence_threshold:
        x = _bicgstab_iteration(
            gradient=gradient,
            self=self,
            preconditioner=preconditioner,
            dirichlet=dirichlet,
            x=x,
            r=r,
            r0=r0,
        )

    outputs = initial_values | x
    outputs = [outputs[k].to_tensor() for k in initial_values.keys()]
    return_vals = tuple(outputs)

    return return_vals


def _bicgstab_iteration(
    gradient: Callable[
        [IPhlowerTensorCollections, list[str], list[str] | None],
        IPhlowerTensorCollections,
    ],
    self: BiCGStabSolver,
    preconditioner: IPreconditioner,
    dirichlet: IPhlowerTensorCollections,
    x: IPhlowerTensorCollections,
    r: IPhlowerTensorCollections,
    r0: IPhlowerTensorCollections,
) -> IPhlowerTensorCollections:
    rho = 1.0
    rho_prev = 1.0
    alpha = 1.0
    beta = 0.0
    omega = 1.0
    v = 0.0

    for itr in range(self._max_iterations):
        self._n_iterated += 1

        rho = self._inner(r.mask(self._update_keys), r0.mask(self._update_keys))
        if rho == 0.0:
            _logger.warning("BiCGStab solver has diverged (rho = 0).")
            self._is_converged = False
            break
        if itr == 0:
            p = phlower_tensor_collection(
                {key: tensor.detach().clone() for key, tensor in r.items()}
            )
        else:
            beta = (rho / rho_prev) * (alpha / omega)
            # p = r + beta(p - omega * AM^-1p(i-1))
            #   = r + beta(p - omega * v)
            ptmp = p.mask(self._update_keys) - omega * v
            p = p | (r.mask(self._update_keys) + beta * ptmp)

        # phat = M^-1p(i-1)
        phat = p | preconditioner.apply(p.mask(self._update_keys))
        # v = AM^-1p(i-1)
        v = gradient(
            phat,
            update_keys=self._update_keys,
            operator_keys=self._operator_keys,
        )
        v = self._apply_dirichlet(v, dirichlet, factor=0.0)
        alpha = rho / self._inner(
            r0.mask(self._update_keys), v.mask(self._update_keys)
        )

        # s(i) = r(i-1) - alpha v
        s = r | (r.mask(self._update_keys) - alpha * v)

        # shat = M^-1 s(i)
        shat = s | preconditioner.apply(s.mask(self._update_keys))

        # t = A * shat
        t = gradient(
            shat,
            update_keys=self._update_keys,
            operator_keys=self._operator_keys,
        )
        t = self._apply_dirichlet(t, dirichlet, factor=0.0)

        # omega = (AM^-1, s) / (AM^-1s, AM^-1s)
        omega = self._inner(
            t.mask(self._update_keys), s.mask(self._update_keys)
        ) / self._inner(t.mask(self._update_keys), t.mask(self._update_keys))

        if omega == 0.0:
            _logger.warning("BiCGStab solver has diverged (omega = 0).")
            self._is_converged = False
            break

        # x(i) = x(i-1) + alpha * M^-1 p(i-1) + omega * M^-1 s(i)
        x = x | (
            x.mask(self._update_keys)
            + alpha * phat.mask(self._update_keys)
            + omega * shat.mask(self._update_keys)
        )

        # r(i) = s(i-1) - omega * AM^-1s(i-1)
        # r = s - omega * t
        r = r | (s.mask(self._update_keys) - omega * t)

        norm_r = self._inner(
            r.mask(self._update_keys), r.mask(self._update_keys)
        )
        criteria = norm_r.apply(torch.linalg.norm)

        if criteria < self._convergence_threshold:
            self._is_converged = True
            break

        if criteria > self._divergence_threshold:
            _logger.warning("BiCGStab solver has diverged.")
            self._is_converged = False
            break

        rho_prev = rho

    if not self._is_converged:
        total_residual = sum(criteria.values()).detach().to("cpu").numpy()
        message = (
            "BiCGStab solver not converged. "
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

    return x


class BiCGStabSolver_Core(torch.autograd.Function):
    clear_saved_tensors_on_access = True

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, *args: tuple
    ) -> tuple[torch.Tensor]:
        num_tensors = len(args) - 4
        tensors = args[:num_tensors]
        initial_values = args[num_tensors]
        problem: IOptimizeProblem = args[num_tensors + 1]
        self: BiCGStabSolver = args[num_tensors + 2]

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

        x = _bicgstab_core(
            problem.gradient,
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
        problem: IOptimizeProblem = ctx.phlower_iteration_problem
        self: BiCGStabSolver = ctx.phlower_gradient_solver
        keys: KeysView[str] = (
            ctx.phlower_iteration_problem_initial_values.keys()
        )
        x_tensors: tuple[torch.Tensor] = ctx.saved_tensors

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

        gradient_rmatvec = partial(rmatvec_core, problem)

        grad_b = _bicgstab_core(
            gradient_rmatvec,
            grad_outputs,
            initial_values_zero,
            self,
            preconditioner=self._preconditioner,
        )

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

        with torch.enable_grad():
            new_x = phlower_tensor_collection(
                dict(zip(keys, x_tensors, strict=True))
            )
            c = problem.gradient(
                new_x,
                update_keys=self._update_keys,
                operator_keys=self._operator_keys,
            )  # The operator is not executing the processing in _bicgstab_core

            tensors = []
            grad_tensors = []
            c_requires_grad = False
            for key, grad_b_tmp in zip(
                self._update_keys, grad_b_core, strict=True
            ):
                c_requires_grad = (
                    c_requires_grad | c[key].to_tensor().requires_grad
                )
                tensors.append(c[key].to_tensor())
                grad_tensors.append(-grad_b_tmp)
            if c_requires_grad:
                torch.autograd.backward(tensors, grad_tensors=grad_tensors)

        return grad_b + tuple(None for _ in range(4))


class BiCGStabSolver(IFIterationSolver):
    @classmethod
    def from_setting(
        cls, setting: IPhlowerIterationSolverSetting
    ) -> BiCGStabSolver:
        assert isinstance(setting, BiCGStabSolverSetting)
        return BiCGStabSolver(**setting.model_dump())

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
            # Be careful that the same preconditioner cannot be applied to
            # both forward and backward passes because A!=A^T

        if self._exact_backward_flag:
            b = phlower_tensor_collection(
                {
                    key: initial_values[self._dict_variable_to_right[key]]
                    if key in self._dict_variable_to_right
                    else torch.zeros_like(initial_values[key])
                    for key in self._update_keys
                }
            )

            outputs = _bicgstab_core(
                problem.gradient,
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
            outputs = BiCGStabSolver_Core.apply(*combined_args)

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
