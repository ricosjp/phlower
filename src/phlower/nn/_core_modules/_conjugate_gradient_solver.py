from __future__ import annotations

import types

import numpy as np
import scipy.sparse as sp
import torch
from numpy.typing import NDArray
from phlower_tensor import ISimulationField, PhlowerTensor, phlower_tensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
)
from scipy.sparse import linalg as lin
from torch.autograd.function import FunctionCtx

from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import ConjugateGradientSolverSetting
from phlower.utils import get_logger

if torch.cuda.is_available():
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse import linalg as clin


_logger = get_logger(__name__)


class ConjugateGradientSolver(torch.nn.Module, IPhlowerCoreModule):
    """
    Solver based on the conjugate gradient (CG) method.

    Parameters
    ----------
    matrix_name: str
        Name of the sparse matrix.
    x0_name: str | None = None
        Name of the initial guess of the solution.
    dirichlet_name: str | None = None
        Name of the Dirichlet feature.
    rtol: float = 1e-5
        Relative tolerance.
    atol: float = 0.0
        Absolute tolerance.
    maxiter: int | None = None
        Maximum number of iterations.
    batch_solve: bool = True
        If True, solve multiple systems in a batched manner.
    force_cpu: bool = False
        If True, force CPU computation even if the input is GPU.
    log_level: str = "warning"
        Log level for the massage when non convergence.

    Examples
    --------
    >>> spmm = ConjugateGradientSolver(matrix_name="A")
    >>> spmm(data, field_data=field)
    """

    @classmethod
    def from_setting(
        cls, setting: ConjugateGradientSolverSetting
    ) -> ConjugateGradientSolver:
        """
        Create ConjugateGradientSolver from ConjugateGradientSolverSetting
        object.

        Args:
            setting (ConjugateGradientSolverSetting):
                setting object for ConjugateGradientSolver

        Returns:
            ConjugateGradientSolver model
        """
        return ConjugateGradientSolver(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "ConjugateGradientSolver"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        matrix_name: str,
        x0_name: str | None = None,
        dirichlet_name: str | None = None,
        rtol: float = 1e-5,
        atol: float = 0.0,
        maxiter: int | None = None,
        batch_solve: bool = True,
        force_cpu: bool = False,
        log_level: str = "warning",
        **kwards,
    ) -> None:
        super().__init__()

        self._matrix_name = matrix_name
        self._x0_name = x0_name
        self._dirichlet_name = dirichlet_name
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._batch_solve = batch_solve
        self._force_cpu = force_cpu
        self._log_level = log_level

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None: ...

    def get_reference_name(self) -> str | None:
        return None

    def _to_tensor_device_if_not_none(
        self, x: PhlowerTensor | None, device: torch.device
    ) -> torch.Tensor | None:
        if x is None:
            return None
        return x.to_tensor().to(device=device)

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField,
        **kwards,
    ) -> PhlowerTensor:
        """forward function which overload torch.nn.Module

        Args:
            data: IPhlowerTensorCollections
                data which receives from predecessors
            field_data: ISimulationField | None
                Constant information through training or prediction

        Returns:
            PhlowerTensor:
                Tensor object
        """
        if self._x0_name is not None and self._x0_name not in data:
            raise KeyError(
                f"x0_name {self._x0_name} not found in {data.keys()}"
            )
        if (
            self._dirichlet_name is not None
            and self._dirichlet_name not in data
        ):
            raise KeyError(
                f"dirichlet_name {self._dirichlet_name} not found in "
                f"{data.keys()}"
            )
        if self._force_cpu:
            device = "cpu"
        else:
            device = None
        x0 = self._to_tensor_device_if_not_none(
            data.pop(self._x0_name, None), device=device
        )
        dirichlet = self._to_tensor_device_if_not_none(
            data.pop(self._dirichlet_name, None), device=device
        )

        b = data.unique_item().to(device=device)
        if b.is_time_series:
            raise NotImplementedError("Time series is not supported")
        matrix = field_data[self._matrix_name].to(device)

        if self._dirichlet_name is not None:
            matrix, b = self._apply_dirichlet(matrix, b, dirichlet)

        solution = SparseCGSolver.apply(
            matrix.to_tensor().to(device),
            b.to_tensor().to(device),
            x0,
            self._rtol,
            self._atol,
            self._maxiter,
            self._batch_solve,
            self._log_level,
        )

        if b.has_dimension:
            ret_dim = b.dimension / matrix.dimension
            ret = phlower_tensor(solution, dimension=ret_dim)
        else:
            ret = phlower_tensor(solution)
        return ret.to(b.device)

    def _apply_dirichlet(
        self, matrix: PhlowerTensor, b: PhlowerTensor, dirichlet: torch.Tensor
    ) -> tuple[PhlowerTensor, PhlowerTensor]:
        n = matrix.shape[0]
        t_mat = matrix.to_tensor().coalesce()

        # Clone to keep b unchanged because we will overwrite reshaped_b
        reshaped_b = torch.reshape(b.to_tensor().clone(), (n, -1))

        reshaped_dirichlet = torch.reshape(dirichlet, (n, -1))
        mask_dirichlet = ~torch.isnan(reshaped_dirichlet[:, 0])
        dirichlet_indices = torch.arange(n, device=b.device)[mask_dirichlet]

        # Move effect of the Dirichlet to RHS (Dirichlet force)
        dirichlet_source = torch.zeros_like(reshaped_b, device=b.device)
        dirichlet_source[mask_dirichlet] = reshaped_dirichlet[mask_dirichlet]
        dirichlet_force = -t_mat @ dirichlet_source
        dirichlet_force[mask_dirichlet] = 0

        # Symmetrically remove off-diagonals for Dirichlet
        values = t_mat.values().clone()
        indices = t_mat.indices()
        row = indices[0]
        col = indices[1]
        mask_row = torch.isin(row, dirichlet_indices)
        mask_col = torch.isin(col, dirichlet_indices)
        mask_diag = row == col
        values[mask_row] = 0.0
        values[mask_col] = 0.0
        values[mask_diag & mask_row] = 1.0

        reshaped_b = reshaped_b + dirichlet_force
        reshaped_b[mask_dirichlet] = reshaped_dirichlet[mask_dirichlet]
        return phlower_tensor(
            torch.sparse_coo_tensor(
                indices=indices, values=values, size=matrix.size()
            ).coalesce(),
            dimension=matrix.dimension,
        ), phlower_tensor(
            torch.reshape(reshaped_b, b.shape), dimension=b.dimension
        )


def to_numpy_or_cupy_array(
    tensor: torch.Tensor,
) -> tuple[types.ModuleType, NDArray]:
    tensor = tensor.detach()

    if tensor.is_cuda:
        arr = cp.asarray(tensor)
        xp = cp
    elif tensor.device.type == "cpu":
        arr = tensor.numpy()
        xp = np
    else:
        raise ValueError(f"Unsupported device type: {tensor.device}")

    return xp, arr


def to_scipy_or_cupy_sparse(
    tensor: torch.Tensor,
) -> tuple[types.ModuleType, sp.sparray | csp.sparray]:
    if not tensor.is_sparse:
        raise ValueError("Input is not suparse")
    if tensor.layout != torch.sparse_coo:
        tensor = tensor.to_sparse_coo()
    tensor = tensor.coalesce()

    shape = tuple(tensor.shape)
    indices = tensor.indices().detach()
    values = tensor.values().detach()

    if tensor.is_cuda:
        row_cp = cp.asarray(indices[0])
        col_cp = cp.asarray(indices[1])
        val_cp = cp.asarray(values)
        sparse_array = csp.coo_matrix((val_cp, (row_cp, col_cp)), shape=shape)
        splin_module = clin

    elif tensor.device.type == "cpu":
        row_np = indices[0].numpy()
        col_np = indices[1].numpy()
        val_np = values.numpy()
        sparse_array = sp.coo_matrix((val_np, (row_np, col_np)), shape=shape)
        splin_module = lin

    else:
        raise ValueError(f"Unsupported device type: {tensor.device}")

    return splin_module, sparse_array


class SparseCGSolver(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        spmat_tensor: torch.Tensor,
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
        rtol: float = 1e-5,
        atol: float = 0.0,
        maxiter: int | None = None,
        batch_solve: bool = True,
        log_level: str = "warning",
    ) -> torch.Tensor:
        if len(b.shape) == 1:
            b = b[:, None]

        xlin, spmat_array = to_scipy_or_cupy_sparse(spmat_tensor)
        xp, b_array = to_numpy_or_cupy_array(b)
        if x0 is None:
            x0_array = None
        else:
            _, x0_array = to_numpy_or_cupy_array(x0)

        x_xp = _solve_multiple(
            xp,
            xlin,
            spmat_array,
            b_array,
            x0=x0_array,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            batch_solve=batch_solve,
            log_level=log_level,
        )
        x = torch.from_dlpack(x_xp).to(b.dtype)

        # Save the sparse components and solution for the backward pass
        ctx.save_for_backward(spmat_tensor.indices(), spmat_tensor.values(), x)
        ctx.spmat_requires_grad = spmat_tensor.requires_grad
        ctx.spmat_size = spmat_tensor.size()
        ctx.xlin = xlin
        ctx.rtol = rtol
        ctx.atol = atol
        ctx.maxiter = maxiter
        ctx.batch_solve = batch_solve
        ctx.log_level = log_level

        return x

    @staticmethod
    def backward(
        ctx: FunctionCtx, grad_x: torch.Tensor
    ) -> tuple[torch.Tensor | None]:
        xp, grad_x_array = to_numpy_or_cupy_array(grad_x)
        indices, values, x = ctx.saved_tensors
        xsp, spmat_array = to_scipy_or_cupy_sparse(
            torch.sparse_coo_tensor(indices, values, size=ctx.spmat_size)
        )

        # Solve the adjoint system A^T grad_b = grad_x,
        # but not need to transpose because A^T = A
        grad_b_array = _solve_multiple(
            xp,
            ctx.xlin,
            spmat_array,
            grad_x_array,
            x0=None,
            rtol=ctx.rtol,
            atol=ctx.atol,
            maxiter=ctx.maxiter,
            batch_solve=ctx.batch_solve,
            log_level=ctx.log_level,
        )
        grad_b = torch.from_dlpack(grad_b_array).to(x.dtype)

        if ctx.spmat_requires_grad:
            # d L / d A_{ij} = - grad_b_i . x_j
            # indices = spmat_tensor.indices().detach()
            row_indices = indices[0]
            col_indices = indices[1]
            grad_spmat_values = -torch.einsum(
                "i...,i...->i", grad_b[row_indices], x[col_indices]
            )
            grad_spmat = torch.sparse_coo_tensor(
                indices, grad_spmat_values, size=ctx.spmat_size
            )
        else:
            grad_spmat = None

        return grad_spmat, grad_b, None, None, None, None, None, None


def _reshape_slice_if_needed(
    arr: NDArray | None,
    shape: tuple[int],
    last_slice: int | slice = slice(None),
) -> NDArray:
    if arr is None:
        return None
    return arr[..., last_slice].reshape(shape)


def _solve_multiple(
    xp: types.ModuleType,
    xlin: types.ModuleType,
    spmat_array: sp.sparray | csp.sparray,
    b_array: NDArray,
    x0: NDArray | None = None,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    batch_solve: bool = True,
    log_level: str = "warning",
) -> np.ndarray | cp.ndarray:
    b_reshaped = xp.reshape(b_array, (spmat_array.shape[0], -1))
    x0_reshaped = _reshape_slice_if_needed(x0, (spmat_array.shape[0], -1))

    if batch_solve:
        n = spmat_array.shape[0]
        n_feature = b_reshaped.shape[-1]

        def matvec(v: NDArray) -> NDArray:
            return (spmat_array @ v.reshape(-1, n_feature)).reshape(-1)

        op = xlin.LinearOperator((n * n_feature, n * n_feature), matvec=matvec)
        x0_input = _reshape_slice_if_needed(x0_reshaped, (-1,))
        x_xp, status = xlin.cg(
            op,
            b_reshaped.reshape(-1),
            x0=x0_input,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
        )
        x_xp = x_xp.reshape(b_array.shape)
    else:
        statuses = []
        list_x_xp = []
        for i in range(b_reshaped.shape[-1]):
            x0_input = _reshape_slice_if_needed(
                x0_reshaped, (-1,), last_slice=i
            )
            _x, status = xlin.cg(
                spmat_array,
                b_reshaped[..., i].reshape(-1),
                x0=x0_input,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
            )
            list_x_xp.append(_x)
            statuses.append(status)
        x_xp = xp.reshape(xp.stack(list_x_xp, axis=-1), b_array.shape)
        status = np.sum(statuses)

    if status != 0:
        message = f"CG solver not converged: {status}"
        if log_level == "warning":
            _logger.warning(message)
        elif log_level == "info":
            _logger.info(message)
        else:
            raise ValueError(f"Invalid log level: {log_level}")

    return x_xp
