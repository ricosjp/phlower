import gc
from functools import partial

import numpy as np
import pytest
import pyvista as pv
import torch
from hypothesis import given
from hypothesis import strategies as st
from phlower_tensor import PhlowerTensor, phlower_tensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower_tensor.functionals import spmm
from scipy import sparse as sp
from scipy.spatial import KDTree

from phlower.nn._group_module import _GroupOptimizeProblem
from phlower.nn._iteration_solvers._bicgstab_solver import (
    BiCGStabSolver,
    rmatvec_core,
)
from phlower.settings._iteration_solver_setting import (
    BiCGStabSolverSetting,
    RandomJacobiCGPreconditionSetting,
)

dtype = torch.float64


class IdentityModel:
    def __init__(self, n_mat: int, coef: torch.Tensor, keys: str = "x"):
        self.n_mat = n_mat
        self.keys = keys

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        sp_array = sp.eye(n_mat).tocoo()
        indices = torch.stack(
            [torch.from_numpy(sp_array.row), torch.from_numpy(sp_array.col)],
            dim=0,
        )
        values = torch.from_numpy(sp_array.data.astype(np.float32))
        values_for_torch = values.clone().requires_grad_(True) + coef
        sparse_for_torch = torch.sparse_coo_tensor(
            indices=indices, values=values_for_torch, size=sp_array.shape
        ).to(self.get_device())

        self.sp_d = phlower_tensor(sparse_for_torch)

    def dot(self, x: IPhlowerTensorCollections) -> IPhlowerTensorCollections:
        return phlower_tensor_collection(
            {k: spmm(self.sp_d, x[k]) for k in self.keys}
        )

    def get_device(self) -> str:
        return self.device


class PoissonModel:
    DISTANCE_FACTOR = 2.1

    def __init__(self, n_x: int, x_length: float = 1.0, keys: str = "x"):
        self.x_length = x_length
        self.keys = keys
        mesh = self._generate_mesh(n_x=n_x)

        self.pos = mesh.points
        self.x = self.pos[:, 0]
        self.dx = self.x[1] - self.x[0]
        self.n = len(self.pos)

        xmin = np.min(self.x)
        xmax = np.max(self.x)
        self.filter_left = np.abs(self.x - xmin) < self.dx * 0.1
        self.filter_right = np.abs(self.x - xmax) < self.dx * 0.1

        self.sp_d = self._generate_laplacian_matrix(mesh)

    def _generate_mesh(self, n_x: int) -> pv.UnstructuredGrid:
        n_points = np.array([n_x, 4, 4])
        grid = pv.ImageData()
        grid.dimensions = n_points

        mesh = grid.cast_to_unstructured_grid()
        mesh.points = mesh.points * self.x_length / np.max(mesh.points[:, 0])
        return mesh

    def _generate_laplacian_matrix(
        self, mesh: pv.UnstructuredGrid
    ) -> PhlowerTensor:
        tree = KDTree(mesh.points)
        distance = self.DISTANCE_FACTOR * self.dx
        sp_d = tree.sparse_distance_matrix(
            tree, max_distance=distance, output_type="coo_matrix"
        )
        sp_d.data = np.exp(-sp_d.data / (distance * 0.1))
        sp_d = sp_d - sp.eye(sp_d.shape[0])
        tot = np.asarray(sp_d.sum(axis=1))
        sp_d = sp_d - sp.eye(sp_d.shape[0]).multiply(tot)

        # Normalize
        mean = np.mean(tot)
        sp_d = (sp_d / mean).tocsr()

        return phlower_tensor(
            torch.sparse_csr_tensor(
                torch.from_numpy(sp_d.indptr),
                torch.from_numpy(sp_d.indices),
                torch.from_numpy(sp_d.data).to(dtype),
                size=sp_d.shape,
            ).to_sparse_coo()
        )

    def laplacian(
        self, x: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        return phlower_tensor_collection(
            {k: spmm(self.sp_d, x[k]) for k in self.keys}
        )


class ConvectiveDiffusion:
    def __init__(self, n_x: int, keys: str = "x"):
        self.keys = keys
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.sp_d = self._generate_matrix(n_x)

    def _generate_matrix(self, n: int) -> PhlowerTensor:

        diag = 4.0 * torch.ones(n, dtype=dtype, device=self.device)
        lower = -2.0 * torch.ones(n - 1, dtype=dtype, device=self.device)
        upper = -1.0 * torch.ones(n - 1, dtype=dtype, device=self.device)

        row = torch.cat(
            [torch.arange(n), torch.arange(1, n), torch.arange(n - 1)]
        )
        col = torch.cat(
            [torch.arange(n), torch.arange(n - 1), torch.arange(1, n)]
        )
        indices = torch.stack([row, col])

        values = torch.cat([diag, lower, upper])

        A = torch.sparse_coo_tensor(
            indices,
            values,
            size=(n, n),
            dtype=dtype,
            device=self.device,
        )

        return phlower_tensor(A.coalesce().to_sparse_coo())

    def dot(self, x: IPhlowerTensorCollections) -> IPhlowerTensorCollections:
        return phlower_tensor_collection(
            {k: spmm(self.sp_d, x[k]) for k in self.keys}
        )

    def get_device(self) -> str:
        return self.device


def test__from_default_setting():
    setting = BiCGStabSolverSetting(
        convergence_threshold=1.0e-4,
        max_iterations=100,
        divergence_threshold=1000,
        update_keys=["a"],
        dict_variable_to_dirichlet={"a": "dirichlet_a"},
    )
    solver = BiCGStabSolver.from_setting(setting)

    assert solver._max_iterations == setting.max_iterations
    assert solver._convergence_threshold == setting.convergence_threshold
    assert solver._divergence_threshold == setting.divergence_threshold
    assert solver._update_keys == setting.update_keys
    assert (
        solver._dict_variable_to_dirichlet == setting.dict_variable_to_dirichlet
    )
    assert solver._precondition_type == "none"


@given(num_of_trials=st.integers(min_value=1, max_value=1000))
def test__from_random_jacobi_setting(num_of_trials: int):
    setting = BiCGStabSolverSetting(
        **{
            "convergence_threshold": 1.0e-4,
            "max_iterations": 100,
            "divergence_threshold": 1000,
            "update_keys": ["a"],
            "dict_variable_to_dirichlet": {"a": "dirichlet_a"},
            "precondition_type": "random_jacobi",
            "precondition_parameters": {
                "num_of_trials": num_of_trials,
            },
        }
    )

    solver = BiCGStabSolver.from_setting(setting)
    assert solver._precondition_type == "random_jacobi"
    assert solver._preconditioner.num_of_trials == num_of_trials


@pytest.mark.parametrize("n_target", [1, 2, 3])
@pytest.mark.parametrize("n_x", [5, 10, 20])
@pytest.mark.parametrize("x_length", [0.1, 1.0, 10.0])
def test__can_converge_laplace_equation(
    n_target: int, n_x: int, x_length: float
):
    operator_keys = [f"x{i}" for i in range(n_target)]
    solver = BiCGStabSolver(
        max_iterations=n_x * 10,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e5,
        update_keys=operator_keys,
        operator_keys=operator_keys,
        dict_variable_to_right={},
        dict_variable_to_dirichlet={
            k: f"x_bnd{i}" for i, k in enumerate(operator_keys)
        },
    )

    poisson_model = PoissonModel(n_x, x_length, keys=operator_keys)
    n = poisson_model.n

    dict_inputs = {}
    dict_inputs.update(
        {
            k: phlower_tensor(torch.ones(n, 1) * 0.1 * i)
            for i, k in enumerate(operator_keys)
        }
    )
    dict_inputs.update(
        {
            right: phlower_tensor(torch.zeros(n, 1))
            for right in solver._dict_variable_to_right.values()
        }
    )
    for i, dirichlet in enumerate(solver._dict_variable_to_dirichlet.values()):
        bnd = torch.ones(n, 1, dtype=dtype) * torch.nan
        bnd[poisson_model.filter_left] = 1.0 * (i + 1)
        bnd[poisson_model.filter_right] = 0.0
        dict_inputs.update({dirichlet: bnd})
    dict_inputs.update({"pos": phlower_tensor(poisson_model.pos)})

    inputs = phlower_tensor_collection(dict_inputs)
    problem = _GroupOptimizeProblem(
        initials=inputs,
        step_forward=poisson_model.laplacian,
        steady_mode=True,
    )
    h = solver.run(inputs, problem)

    for i, target in enumerate(operator_keys):
        actual = h[target]
        desired = (x_length - poisson_model.pos[:, [0]]) / x_length * (i + 1)
        scale = np.max(desired)
        np.testing.assert_array_almost_equal(
            actual.numpy() / scale, desired / scale, decimal=2
        )


@pytest.mark.parametrize("n_x", [5, 10, 20])
@pytest.mark.parametrize("x_length", [0.1, 1.0, 10.0])
def test__can_converge_poisson_equation(n_x: int, x_length: float):
    solver = BiCGStabSolver(
        max_iterations=n_x * 10,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e5,
        update_keys=["x"],
        operator_keys=["x"],
        dict_variable_to_right={"x": "b"},
        dict_variable_to_dirichlet={"x": "x_bnd"},
    )

    poisson_model = PoissonModel(n_x, x_length=x_length)
    n = poisson_model.n
    x_bnd = torch.ones(n, 1, dtype=dtype) * torch.nan
    x_bnd[poisson_model.filter_left] = 0.0
    x_bnd[poisson_model.filter_right] = 0.0
    b = torch.ones(n, 1, dtype=dtype) * poisson_model.dx**2 / 2
    inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(torch.zeros(n, 1, dtype=dtype)),
            "b": phlower_tensor(b),
            "x_bnd": phlower_tensor(x_bnd),
            "pos": phlower_tensor(poisson_model.pos),
        }
    )
    problem = _GroupOptimizeProblem(
        initials=inputs,
        step_forward=poisson_model.laplacian,
        steady_mode=True,
    )
    h = solver.run(inputs, problem)

    actual = h.unique_item()
    x = poisson_model.x[:, None]
    desired = -x * (x - x_length)
    scale = np.max(desired)
    np.testing.assert_array_almost_equal(
        actual / scale, desired / scale, decimal=1
    )


@pytest.mark.parametrize("n_x", [5, 10, 20])
def test__can_converge_convective_diffusion_equation(n_x: int):
    solver = BiCGStabSolver(
        max_iterations=n_x * 10,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e5,
        update_keys=["x"],
        operator_keys=["x"],
        dict_variable_to_right={"x": "b"},
        dict_variable_to_dirichlet={"x": "x_bnd"},
    )

    convective_diffusion_model = ConvectiveDiffusion(n_x=n_x)
    device = convective_diffusion_model.get_device()

    x_bnd = torch.ones(n_x, 1, dtype=dtype, device=device) * torch.nan
    # notice: not Ax=b, but Ax+b=0
    b = -torch.ones(n_x, 1, dtype=dtype, device=device)
    b[0:1] = -3
    b[n_x - 1 : n_x] = -2
    inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(
                torch.zeros(n_x, 1, dtype=dtype, device=device)
            ),
            "b": phlower_tensor(b),
            "x_bnd": phlower_tensor(x_bnd),
        }
    )
    problem = _GroupOptimizeProblem(
        initials=inputs,
        step_forward=convective_diffusion_model.dot,
        steady_mode=True,
    )
    h = solver.run(inputs, problem)

    actual = h.unique_item()
    desired = np.ones((n_x, 1))
    scale = np.max(desired)
    np.testing.assert_array_almost_equal(
        actual.to("cpu") / scale, desired / scale, decimal=1
    )


@pytest.mark.parametrize("n_x", [5])
@pytest.mark.parametrize("x_length", [1.0])
def test__can_run_multiple_inputs(n_x: int, x_length: float):
    solver = BiCGStabSolver(
        max_iterations=n_x * 100,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e5,
        update_keys=["x"],
        operator_keys=["x"],
        dict_variable_to_right={"x": "b"},
        dict_variable_to_dirichlet={"x": "x_bnd"},
    )

    poisson_model = PoissonModel(n_x, x_length=x_length)
    n = poisson_model.n
    x_bnd = torch.ones(n, 1, dtype=dtype) * torch.nan
    x_bnd[poisson_model.filter_left] = 0.0
    x_bnd[poisson_model.filter_right] = 0.0
    b = torch.ones(n, 1, dtype=dtype) * poisson_model.dx**2 / 2
    inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(torch.zeros(n, 1, dtype=dtype)),
            "b": phlower_tensor(b),
            "nu": phlower_tensor(torch.ones(1, 1, dtype=dtype) * 0.95),
            "x_bnd": phlower_tensor(x_bnd),
            "pos": phlower_tensor(poisson_model.pos),
        }
    )

    def forward(x: IPhlowerTensorCollections) -> IPhlowerTensorCollections:
        lap = poisson_model.laplacian(x)
        lap.update({"x": x["nu"] * lap["x"]}, overwrite=True)
        return lap

    problem = _GroupOptimizeProblem(
        initials=inputs,
        step_forward=forward,
        steady_mode=True,
    )
    h = solver.run(inputs, problem)

    actual = h.unique_item()
    x = poisson_model.x[:, None]
    desired = -x * (x - x_length)
    scale = np.max(desired)
    np.testing.assert_array_almost_equal(
        actual / scale, desired / scale, decimal=1
    )


def test__raises_value_error_when_right_keys_incomplete():
    with pytest.raises(ValueError, match="Unmatched dict_variable_to_right"):
        BiCGStabSolver(
            max_iterations=10,
            convergence_threshold=1.0e-8,
            divergence_threshold=1.0e5,
            update_keys=["x", "y"],
            operator_keys=["x", "y"],
            dict_variable_to_right={"z": "b"},
            dict_variable_to_dirichlet={"x": "x_bnd", "y": "y_bnd"},
        )


def test__raises_value_error_when_dirichlet_keys_incomplete():
    with pytest.raises(
        ValueError, match="Unmatched dict_variable_to_dirichlet"
    ):
        BiCGStabSolver(
            max_iterations=10,
            convergence_threshold=1.0e-8,
            divergence_threshold=1.0e5,
            update_keys=["x", "y"],
            operator_keys=["x", "y"],
            dict_variable_to_right={"x": "b", "y": "c"},
            dict_variable_to_dirichlet={"z": "x_bnd"},
        )


@pytest.mark.parametrize("n_x", [5])
@pytest.mark.parametrize("x_length", [0.1])
def test__rmatvec(n_x: int, x_length: float):
    solver = BiCGStabSolver(
        max_iterations=n_x * 10,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e5,
        update_keys=["x"],
        operator_keys=["x"],
        dict_variable_to_right={"x": "b"},
        dict_variable_to_dirichlet={"x": "x_bnd"},
    )
    poisson_model = PoissonModel(n_x, x_length=x_length)

    n = poisson_model.n
    x_bnd = torch.ones(n, 1, dtype=dtype) * torch.nan
    x_bnd[poisson_model.filter_left] = 0.0
    x_bnd[poisson_model.filter_right] = 0.0
    b = torch.ones(n, 1, dtype=dtype) * poisson_model.dx**2 / 2

    inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(torch.randn(n, 1, dtype=dtype)),
            "b": phlower_tensor(b),
            "x_bnd": phlower_tensor(x_bnd),
            "pos": phlower_tensor(poisson_model.pos),
        }
    )

    problem = _GroupOptimizeProblem(
        initials=inputs,
        step_forward=poisson_model.laplacian,
        steady_mode=True,
    )

    matvec_result = problem.gradient(
        inputs,
        update_keys=solver._update_keys,
        operator_keys=solver._operator_keys,
    )

    gradient_rmatvec = partial(rmatvec_core, problem)

    rmatvec_result = gradient_rmatvec(
        inputs,
        update_keys=solver._update_keys,
        operator_keys=solver._operator_keys,
    )

    a1 = matvec_result.unique_item()
    b1 = rmatvec_result.unique_item()
    scale = np.max(a1)
    np.testing.assert_array_almost_equal(a1 / scale, b1 / scale, decimal=4)

    def forward(x: IPhlowerTensorCollections) -> IPhlowerTensorCollections:
        tmp = x["x"]._tensor
        perm = torch.arange(tmp.size(0), device=tmp.device)
        perm[0] = 1
        perm[1] = 0
        x_new = tmp[perm]
        sign = torch.ones_like(x_new)
        sign[0] = -1
        x_new2 = x_new * sign
        z = x.clone()
        z.update({"x": phlower_tensor(x_new2)}, overwrite=True)
        return z

    inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(torch.randn(4, 1)),
        }
    )

    problem = _GroupOptimizeProblem(
        initials=inputs,
        step_forward=forward,
        steady_mode=True,
    )

    matvec_result = problem.gradient(
        inputs,
        update_keys=solver._update_keys,
        operator_keys=solver._operator_keys,
    )

    gradient_rmatvec = partial(rmatvec_core, problem)

    rmatvec_result = gradient_rmatvec(
        inputs,
        update_keys=solver._update_keys,
        operator_keys=solver._operator_keys,
    )

    a1 = matvec_result.unique_item()
    b1 = rmatvec_result.unique_item()
    scale = np.max(a1)
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(a1 / scale, b1 / scale, decimal=4)


@pytest.mark.parametrize("n_x", [5, 10])
@pytest.mark.parametrize("x_length", [1.0, 10.0])
def test__check_gradient_value(n_x: int, x_length: float):
    solver = BiCGStabSolver(
        max_iterations=n_x * 10,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e8,
        update_keys=["x"],
        operator_keys=["x"],
        dict_variable_to_right={"x": "b"},
        dict_variable_to_dirichlet={"x": "x_bnd"},
    )
    low_memory_solver = BiCGStabSolver(
        max_iterations=n_x * 10,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e8,
        update_keys=["x"],
        operator_keys=["x"],
        dict_variable_to_right={"x": "b"},
        dict_variable_to_dirichlet={"x": "x_bnd"},
        exact_backward_flag=False,
    )

    coef1 = torch.tensor(
        [
            0.3,
        ],
        requires_grad=True,
        dtype=dtype,
    )
    low_memory_coef1 = torch.tensor(
        [
            0.3,
        ],
        requires_grad=True,
        dtype=dtype,
    )
    poisson_model = PoissonModel(n_x, x_length=x_length)
    indices = poisson_model.sp_d._tensor.indices()
    shape = poisson_model.sp_d._tensor.shape

    new_values = poisson_model.sp_d._tensor.values() + coef1
    new_tensor = torch.sparse_coo_tensor(indices, new_values, shape)
    poisson_model.sp_d._tensor = new_tensor

    low_memory_poisson_model = PoissonModel(n_x, x_length=x_length)
    low_memory_new_values = (
        low_memory_poisson_model.sp_d._tensor.values() + low_memory_coef1
    )
    low_memory_new_tensor = torch.sparse_coo_tensor(
        indices, low_memory_new_values, shape
    )
    low_memory_poisson_model.sp_d._tensor = low_memory_new_tensor

    n = poisson_model.n
    x_bnd = torch.ones(n, 1, dtype=dtype) * torch.nan
    x_bnd[poisson_model.filter_left] = 0.0
    x_bnd[poisson_model.filter_right] = 0.0
    b = torch.ones(n, 1, dtype=dtype) * poisson_model.dx**2 / 2

    coef2 = torch.tensor(
        [
            0.1,
        ],
        requires_grad=True,
        dtype=dtype,
    )
    low_memory_coef2 = torch.tensor(
        [
            0.1,
        ],
        requires_grad=True,
        dtype=dtype,
    )

    inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(torch.zeros(n, 1, dtype=dtype)),
            "b": phlower_tensor(b + coef2),
            "x_bnd": phlower_tensor(x_bnd),
            "pos": phlower_tensor(poisson_model.pos),
        }
    )

    low_memory_inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(torch.zeros(n, 1, dtype=dtype)),
            "b": phlower_tensor(b + low_memory_coef2),
            "x_bnd": phlower_tensor(x_bnd),
            "pos": phlower_tensor(poisson_model.pos),
        }
    )

    problem = _GroupOptimizeProblem(
        initials=inputs,
        step_forward=poisson_model.laplacian,
        steady_mode=True,
    )
    low_memory_problem = _GroupOptimizeProblem(
        initials=low_memory_inputs,
        step_forward=low_memory_poisson_model.laplacian,
        steady_mode=True,
    )
    h = solver.run(inputs, problem)
    low_memory_h = low_memory_solver.run(low_memory_inputs, low_memory_problem)

    x = poisson_model.x[:, None]
    desired = -x * (x - x_length)
    scale = np.max(desired)

    diff = (
        h.unique_item() / scale - torch.from_numpy(np.asarray(desired)) / scale
    )
    low_memory_diff = (
        low_memory_h.unique_item() / scale
        - torch.from_numpy(np.asarray(desired)) / scale
    )

    loss = torch.linalg.norm(diff)
    low_memory_loss = torch.linalg.norm(low_memory_diff)

    scale = loss.numpy()
    np.testing.assert_array_almost_equal(
        loss.numpy() / scale, low_memory_loss.numpy() / scale, decimal=4
    )

    loss.backward()
    low_memory_loss.backward()

    scale = np.max(coef1.grad.numpy())
    np.testing.assert_array_almost_equal(
        coef1.grad.numpy() / scale,
        low_memory_coef1.grad.numpy() / scale,
        decimal=2,
    )
    scale = np.max(coef2.grad.numpy())
    np.testing.assert_array_almost_equal(
        coef2.grad.numpy() / scale,
        low_memory_coef2.grad.numpy() / scale,
        decimal=3,
    )

    """
    # for gradcheck

    low_memory_poisson_model.sp_d._tensor = (
        low_memory_poisson_model.sp_d._tensor.to(torch.float64)
    )

    tensors = [
        low_memory_inputs[key]._tensor.requires_grad_(True).to(torch.float64)
        for key in low_memory_inputs.keys()
    ]
    dummy_tensor = torch.tensor(0.0, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda *ts: BiCGStabSolver_Core.apply(
            *(
                ts
                + (low_memory_inputs,)
                + (low_memory_problem,)
                + (low_memory_solver,)
                + (dummy_tensor,)
            )
        ),
        (
            tuple(
                tensors,
            )
        ),
    )
    """


@pytest.mark.parametrize("n_x", [5, 10, 20])
def test__check_asymmetric_gradient_value(n_x: int):
    solver = BiCGStabSolver(
        max_iterations=n_x * 10,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e5,
        update_keys=["x"],
        operator_keys=["x"],
        dict_variable_to_right={"x": "b"},
        dict_variable_to_dirichlet={"x": "x_bnd"},
    )
    low_memory_solver = BiCGStabSolver(
        max_iterations=n_x * 10,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e5,
        update_keys=["x"],
        operator_keys=["x"],
        dict_variable_to_right={"x": "b"},
        dict_variable_to_dirichlet={"x": "x_bnd"},
        exact_backward_flag=False,
    )

    convective_diffusion_model = ConvectiveDiffusion(n_x=n_x)
    low_memory_convective_diffusion_model = ConvectiveDiffusion(n_x=n_x)
    device = convective_diffusion_model.get_device()

    coef1 = torch.tensor(
        [
            0.3,
        ],
        requires_grad=True,
        dtype=dtype,
        device=device,
    )
    low_memory_coef1 = torch.tensor(
        [
            0.3,
        ],
        requires_grad=True,
        dtype=dtype,
        device=device,
    )
    indices = convective_diffusion_model.sp_d._tensor.indices()
    shape = convective_diffusion_model.sp_d._tensor.shape

    new_values = convective_diffusion_model.sp_d._tensor.values() + coef1
    new_tensor = torch.sparse_coo_tensor(indices, new_values, shape)
    convective_diffusion_model.sp_d._tensor = new_tensor

    low_memory_new_values = (
        low_memory_convective_diffusion_model.sp_d._tensor.values()
        + low_memory_coef1
    )
    low_memory_new_tensor = torch.sparse_coo_tensor(
        indices, low_memory_new_values, shape
    )
    low_memory_convective_diffusion_model.sp_d._tensor = low_memory_new_tensor

    x_bnd = torch.ones(n_x, 1, dtype=dtype, device=device) * torch.nan
    b = -torch.ones(n_x, 1, dtype=dtype, device=device)
    b[0:1] = -3
    b[n_x - 1 : n_x] = -2

    coef2 = torch.tensor(
        [
            0.1,
        ],
        requires_grad=True,
        dtype=dtype,
        device=device,
    )
    low_memory_coef2 = torch.tensor(
        [
            0.1,
        ],
        requires_grad=True,
        dtype=dtype,
        device=device,
    )

    inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(
                torch.zeros(n_x, 1, dtype=dtype, device=device)
            ),
            "b": phlower_tensor(b + coef2),
            "x_bnd": phlower_tensor(x_bnd),
        }
    )

    low_memory_inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(
                torch.zeros(n_x, 1, dtype=dtype, device=device)
            ),
            "b": phlower_tensor(b + low_memory_coef2),
            "x_bnd": phlower_tensor(x_bnd),
        }
    )

    problem = _GroupOptimizeProblem(
        initials=inputs,
        step_forward=convective_diffusion_model.dot,
        steady_mode=True,
    )
    low_memory_problem = _GroupOptimizeProblem(
        initials=low_memory_inputs,
        step_forward=low_memory_convective_diffusion_model.dot,
        steady_mode=True,
    )
    h = solver.run(inputs, problem)
    low_memory_h = low_memory_solver.run(low_memory_inputs, low_memory_problem)

    desired = np.ones((n_x, 1))
    scale = np.max(desired)

    diff = (
        h.unique_item() / scale
        - torch.from_numpy(np.asarray(desired)).to(device=device) / scale
    )
    low_memory_diff = (
        low_memory_h.unique_item() / scale
        - torch.from_numpy(np.asarray(desired)).to(device=device) / scale
    )

    loss = torch.linalg.norm(diff)
    low_memory_loss = torch.linalg.norm(low_memory_diff)

    scale = loss.numpy()
    np.testing.assert_array_almost_equal(
        loss.numpy() / scale, low_memory_loss.numpy() / scale, decimal=4
    )

    loss.backward()
    low_memory_loss.backward()

    scale = np.max(coef1.grad.to("cpu").numpy())
    np.testing.assert_array_almost_equal(
        coef1.grad.to("cpu").numpy() / scale,
        low_memory_coef1.grad.to("cpu").numpy() / scale,
        decimal=2,
    )
    scale = np.max(coef2.grad.to("cpu").numpy())
    np.testing.assert_array_almost_equal(
        coef2.grad.to("cpu").numpy() / scale,
        low_memory_coef2.grad.to("cpu").numpy() / scale,
        decimal=3,
    )

    """
    # for gradcheck

    low_memory_convective_diffusion_model.sp_d._tensor = (
        low_memory_convective_diffusion_model.sp_d._tensor.to(torch.float64)
    )

    tensors = [
        low_memory_inputs[key]._tensor.requires_grad_(True).to(torch.float64)
        for key in low_memory_inputs.keys()
    ]
    dummy_tensor = torch.tensor(0.0, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda *ts: BiCGStabSolver_Core.apply(
            *(
                ts
                + (low_memory_inputs,)
                + (low_memory_problem,)
                + (low_memory_solver,)
                + (dummy_tensor,)
            )
        ),
        (
            tuple(
                tensors,
            )
        ),
    )
    """


@pytest.mark.gpu_test
@pytest.mark.parametrize("n_mat", [10000])
def test__check_memory_usage(n_mat: int):
    solver = BiCGStabSolver(
        max_iterations=2,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e5,
        update_keys=["x"],
        operator_keys=["x"],
        dict_variable_to_right={"x": "b"},
        dict_variable_to_dirichlet={"x": "x_bnd"},
    )
    low_memory_solver = BiCGStabSolver(
        max_iterations=2,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e5,
        update_keys=["x"],
        operator_keys=["x"],
        dict_variable_to_right={"x": "b"},
        dict_variable_to_dirichlet={"x": "x_bnd"},
        exact_backward_flag=False,
    )

    convective_diffusion_model = ConvectiveDiffusion(n_x=n_mat)
    low_memory_convective_diffusion_model = ConvectiveDiffusion(n_x=n_mat)

    device = convective_diffusion_model.get_device()

    coef1 = torch.tensor(
        [
            1.0,
        ],
        requires_grad=True,
        dtype=dtype,
        device=device,
    )
    coef2 = torch.tensor(
        [
            2.0,
        ],
        requires_grad=True,
        dtype=dtype,
        device=device,
    )
    low_memory_coef1 = torch.tensor(
        [
            1.0,
        ],
        requires_grad=True,
        dtype=dtype,
        device=device,
    )
    low_memory_coef2 = torch.tensor(
        [
            2.0,
        ],
        requires_grad=True,
        dtype=dtype,
        device=device,
    )

    indices = convective_diffusion_model.sp_d._tensor.indices()
    shape = convective_diffusion_model.sp_d._tensor.shape
    new_values = convective_diffusion_model.sp_d._tensor.values() + coef1
    new_tensor = torch.sparse_coo_tensor(indices, new_values, shape)
    convective_diffusion_model.sp_d._tensor = new_tensor

    low_memory_new_values = (
        low_memory_convective_diffusion_model.sp_d._tensor.values()
        + low_memory_coef1
    )
    low_memory_new_tensor = torch.sparse_coo_tensor(
        indices, low_memory_new_values, shape
    )
    low_memory_convective_diffusion_model.sp_d._tensor = low_memory_new_tensor

    x_bnd = torch.ones(n_mat, 1, dtype=dtype, device=device) * torch.nan
    b = torch.ones(n_mat, 1, dtype=dtype, device=device)
    b[0:1] = -3
    b[n_mat - 1 : n_mat] = -2

    inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(
                torch.zeros(n_mat, 1, dtype=dtype, device=device)
            ),
            "b": phlower_tensor(b + coef2),
            "x_bnd": phlower_tensor(x_bnd),
        }
    ).to(convective_diffusion_model.get_device())
    low_memory_inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(
                torch.zeros(n_mat, 1, dtype=dtype, device=device)
            ),
            "b": phlower_tensor(b + low_memory_coef2),
            "x_bnd": phlower_tensor(x_bnd),
        }
    ).to(low_memory_convective_diffusion_model.get_device())

    problem = _GroupOptimizeProblem(
        initials=inputs,
        step_forward=convective_diffusion_model.dot,
        steady_mode=True,
    )
    low_memory_problem = _GroupOptimizeProblem(
        initials=low_memory_inputs,
        step_forward=low_memory_convective_diffusion_model.dot,
        steady_mode=True,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    low_memory_h = low_memory_solver.run(low_memory_inputs, low_memory_problem)
    low_memory_loss = torch.sum(low_memory_h.unique_item())
    low_memory_loss.backward()

    low_memory_max_allocated = torch.cuda.max_memory_allocated() / 1024**2

    del low_memory_h, low_memory_loss

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    h = solver.run(inputs, problem)
    loss = torch.sum(h.unique_item())
    loss.backward()

    max_allocated = torch.cuda.max_memory_allocated() / 1024**2

    del h, loss

    assert (
        low_memory_max_allocated * 1.05 < max_allocated
        or convective_diffusion_model.get_device() == "cpu"
    )


def test__raise_value_error_when_unsupported_precondition():
    with pytest.raises(
        NotImplementedError, match="precondition_type: hoge is not implemented."
    ):
        _ = BiCGStabSolverSetting(
            convergence_threshold=1.0e-4,
            max_iterations=100,
            divergence_threshold=1000,
            update_keys=["a"],
            dict_variable_to_dirichlet={"a": "dirichlet_a"},
            precondition_type="hoge",
        )


@pytest.mark.parametrize("n_target", [3])
@pytest.mark.parametrize("n_x", [5])
@pytest.mark.parametrize("x_length", [0.1])
def test__can_converge_laplace_equation_with_RandomJacobi_precond(
    n_target: int, n_x: int, x_length: float
):
    torch.manual_seed(0)
    operator_keys = [f"x{i}" for i in range(n_target)]
    solver = BiCGStabSolver(
        max_iterations=n_x * 10,
        convergence_threshold=1.0e-6,
        divergence_threshold=1.0e8,
        update_keys=operator_keys,
        operator_keys=operator_keys,
        dict_variable_to_right={},
        dict_variable_to_dirichlet={
            k: f"x_bnd{i}" for i, k in enumerate(operator_keys)
        },
        precondition_type="random_jacobi",
        precondition_parameters=RandomJacobiCGPreconditionSetting(
            num_of_trials=50
        ),
    )

    poisson_model = PoissonModel(n_x, x_length, keys=operator_keys)
    n = poisson_model.n

    dict_inputs = {}
    dict_inputs.update(
        {
            k: phlower_tensor(torch.ones(n, 1, dtype=dtype) * 0.1 * i)
            for i, k in enumerate(operator_keys)
        }
    )
    dict_inputs.update(
        {
            right: phlower_tensor(torch.zeros(n, 1, dtype=dtype))
            for right in solver._dict_variable_to_right.values()
        }
    )
    for i, dirichlet in enumerate(solver._dict_variable_to_dirichlet.values()):
        bnd = torch.ones(n, 1, dtype=dtype) * torch.nan
        bnd[poisson_model.filter_left] = 1.0 * (i + 1)
        bnd[poisson_model.filter_right] = 0.0
        dict_inputs.update({dirichlet: bnd})
    dict_inputs.update({"pos": phlower_tensor(poisson_model.pos)})

    inputs = phlower_tensor_collection(dict_inputs)
    problem = _GroupOptimizeProblem(
        initials=inputs,
        step_forward=poisson_model.laplacian,
        steady_mode=True,
    )
    h = solver.run(inputs, problem)

    for i, target in enumerate(operator_keys):
        actual = h[target]
        desired = (x_length - poisson_model.pos[:, [0]]) / x_length * (i + 1)
        scale = np.max(desired)
        np.testing.assert_array_almost_equal(
            actual.numpy() / scale, desired / scale, decimal=2
        )


@pytest.mark.parametrize("n_target", [1])
@pytest.mark.parametrize("n_x", [200])
@pytest.mark.parametrize("x_length", [0.1])
def test__check_iterations_with_precond(
    n_target: int, n_x: int, x_length: float
):
    torch.manual_seed(0)
    operator_keys = [f"x{i}" for i in range(n_target)]
    solver = BiCGStabSolver(
        max_iterations=n_x * 10,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e8,
        update_keys=operator_keys,
        operator_keys=operator_keys,
        dict_variable_to_right={},
        dict_variable_to_dirichlet={
            k: f"x_bnd{i}" for i, k in enumerate(operator_keys)
        },
    )
    RandomJacobisolver = BiCGStabSolver(
        max_iterations=n_x * 10,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e8,
        update_keys=operator_keys,
        operator_keys=operator_keys,
        dict_variable_to_right={},
        dict_variable_to_dirichlet={
            k: f"x_bnd{i}" for i, k in enumerate(operator_keys)
        },
        precondition_type="random_jacobi",
        precondition_parameters=RandomJacobiCGPreconditionSetting(
            num_of_trials=3000
        ),
    )

    poisson_model = PoissonModel(n_x, x_length, keys=operator_keys)
    n = poisson_model.n

    dict_inputs = {}
    dict_inputs.update(
        {
            k: phlower_tensor(torch.ones(n, 1, dtype=dtype) * 0.1 * i)
            for i, k in enumerate(operator_keys)
        }
    )
    dict_inputs.update(
        {
            right: phlower_tensor(torch.zeros(n, 1, dtype=dtype))
            for right in solver._dict_variable_to_right.values()
        }
    )
    for i, dirichlet in enumerate(solver._dict_variable_to_dirichlet.values()):
        bnd = torch.ones(n, 1, dtype=dtype) * torch.nan
        bnd[poisson_model.filter_left] = 1.0 * (i + 1)
        bnd[poisson_model.filter_right] = 0.0
        dict_inputs.update({dirichlet: bnd})
    dict_inputs.update({"pos": phlower_tensor(poisson_model.pos)})

    inputs = phlower_tensor_collection(dict_inputs)
    problem = _GroupOptimizeProblem(
        initials=inputs,
        step_forward=poisson_model.laplacian,
        steady_mode=True,
    )
    _ = solver.run(inputs, problem)
    _ = RandomJacobisolver.run(inputs, problem)

    assert RandomJacobisolver._n_iterated < solver._n_iterated
