import numpy as np
import pytest
import pyvista as pv
import torch
from phlower_tensor import PhlowerTensor, phlower_tensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower_tensor.functionals import spmm
from scipy import sparse as sp
from scipy.spatial import KDTree

from phlower.nn._group_module import _GroupOptimizeProblem
from phlower.nn._iteration_solvers._conjugate_gradient_solver import (
    ConjugateGradientSolver,
)
from phlower.settings._iteration_solver_setting import (
    ConjugateGradientSolverSetting,
)


class IdentityModel:
    def __init__(self, n_mat: int, coef: torch.Tensor, keys: str = "x"):
        self.n_mat = n_mat
        self.keys = keys

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        sp_array = sp.eye(n_mat, dtype=np.float32).tocoo()
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
                torch.from_numpy(sp_d.data).to(torch.float32),
                size=sp_d.shape,
            ).to_sparse_coo()
        )

    def laplacian(
        self, x: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        return phlower_tensor_collection(
            {k: spmm(self.sp_d, x[k]) for k in self.keys}
        )


def test__from_setting():
    setting = ConjugateGradientSolverSetting(
        convergence_threshold=1.0e-4,
        max_iterations=100,
        divergence_threshold=1000,
        update_keys=["a"],
        dict_variable_to_dirichlet={"a": "dirichlet_a"},
    )
    solver = ConjugateGradientSolver.from_setting(setting)
    for key, value in vars(setting).items():
        actual = getattr(solver, "_" + key)
        if isinstance(value, float):
            np.testing.assert_almost_equal(actual, value)
        else:
            assert actual == value


@pytest.mark.parametrize("n_target", [1, 2, 3])
@pytest.mark.parametrize("n_x", [5, 10, 20])
@pytest.mark.parametrize("x_length", [0.1, 1.0, 10.0])
def test__can_converge_laplace_equation(
    n_target: int, n_x: int, x_length: float
):
    operator_keys = [f"x{i}" for i in range(n_target)]
    solver = ConjugateGradientSolver(
        max_iterations=n_x * 10,
        convergence_threshold=1.0e-5,
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
        bnd = torch.ones(n, 1) * torch.nan
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
    solver = ConjugateGradientSolver(
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
    x_bnd = torch.ones(n, 1) * torch.nan
    x_bnd[poisson_model.filter_left] = 0.0
    x_bnd[poisson_model.filter_right] = 0.0
    b = torch.ones(n, 1) * poisson_model.dx**2 / 2
    inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(torch.zeros(n, 1)),
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


def test__raises_value_error_when_right_keys_incomplete():
    with pytest.raises(ValueError, match="Unmatched dict_variable_to_right"):
        ConjugateGradientSolver(
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
        ConjugateGradientSolver(
            max_iterations=10,
            convergence_threshold=1.0e-8,
            divergence_threshold=1.0e5,
            update_keys=["x", "y"],
            operator_keys=["x", "y"],
            dict_variable_to_right={"x": "b", "y": "c"},
            dict_variable_to_dirichlet={"z": "x_bnd"},
        )


@pytest.mark.parametrize("n_x", [5, 10])
@pytest.mark.parametrize("x_length", [0.1, 1.0])
def test__check_gradient_value(n_x: int, x_length: float):
    solver = ConjugateGradientSolver(
        max_iterations=n_x * 10,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e5,
        update_keys=["x"],
        operator_keys=["x"],
        dict_variable_to_right={"x": "b"},
        dict_variable_to_dirichlet={"x": "x_bnd"},
    )
    low_memory_solver = ConjugateGradientSolver(
        max_iterations=n_x * 10,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e5,
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
    )
    low_memory_coef1 = torch.tensor(
        [
            0.3,
        ],
        requires_grad=True,
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
    x_bnd = torch.ones(n, 1) * torch.nan
    x_bnd[poisson_model.filter_left] = 0.0
    x_bnd[poisson_model.filter_right] = 0.0
    b = torch.ones(n, 1) * poisson_model.dx**2 / 2

    coef2 = torch.tensor(
        [
            0.1,
        ],
        requires_grad=True,
    )
    low_memory_coef2 = torch.tensor(
        [
            0.1,
        ],
        requires_grad=True,
    )

    inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(torch.zeros(n, 1)),
            "b": phlower_tensor(b + coef2),
            "x_bnd": phlower_tensor(x_bnd),
            "pos": phlower_tensor(poisson_model.pos),
        }
    )

    low_memory_inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(torch.zeros(n, 1)),
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
        decimal=4,
    )
    scale = np.max(coef2.grad.numpy())
    np.testing.assert_array_almost_equal(
        coef2.grad.numpy() / scale,
        low_memory_coef2.grad.numpy() / scale,
        decimal=4,
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
        lambda *ts: ConjugateGradientSolver_Core.apply(
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
@pytest.mark.parametrize("n_mat", [1000000])
def test__check_memory_usage(n_mat: int):
    solver = ConjugateGradientSolver(
        max_iterations=2,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e5,
        update_keys=["x"],
        operator_keys=["x"],
        dict_variable_to_right={"x": "b"},
        dict_variable_to_dirichlet={"x": "x_bnd"},
    )
    low_memory_solver = ConjugateGradientSolver(
        max_iterations=2,
        convergence_threshold=1.0e-8,
        divergence_threshold=1.0e5,
        update_keys=["x"],
        operator_keys=["x"],
        dict_variable_to_right={"x": "b"},
        dict_variable_to_dirichlet={"x": "x_bnd"},
        exact_backward_flag=False,
    )

    coef1 = torch.tensor(
        [
            1.0,
        ],
        requires_grad=True,
    )
    coef2 = torch.tensor(
        [
            2.0,
        ],
        requires_grad=True,
    )
    low_memory_coef1 = torch.tensor(
        [
            1.0,
        ],
        requires_grad=True,
    )
    low_memory_coef2 = torch.tensor(
        [
            2.0,
        ],
        requires_grad=True,
    )

    identity_model = IdentityModel(n_mat, coef1)
    low_memory_identity_model = IdentityModel(n_mat, low_memory_coef1)

    x_bnd = torch.ones(n_mat, 1) * torch.nan
    b = torch.ones(n_mat, 1)

    inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(torch.zeros(n_mat, 1)),
            "b": phlower_tensor(b + coef2),
            "x_bnd": phlower_tensor(x_bnd),
        }
    ).to(identity_model.get_device())
    low_memory_inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(torch.zeros(n_mat, 1)),
            "b": phlower_tensor(b + low_memory_coef2),
            "x_bnd": phlower_tensor(x_bnd),
        }
    ).to(low_memory_identity_model.get_device())

    problem = _GroupOptimizeProblem(
        initials=inputs,
        step_forward=identity_model.dot,
        steady_mode=True,
    )
    low_memory_problem = _GroupOptimizeProblem(
        initials=low_memory_inputs,
        step_forward=low_memory_identity_model.dot,
        steady_mode=True,
    )

    torch.cuda.empty_cache()
    low_memory_h = low_memory_solver.run(low_memory_inputs, low_memory_problem)
    low_memory_loss = torch.sum(low_memory_h.unique_item())
    low_memory_loss.backward()

    low_memory_max_allocated = torch.cuda.max_memory_allocated() / 1024**2

    del low_memory_h, low_memory_loss

    torch.cuda.empty_cache()
    h = solver.run(inputs, problem)
    loss = torch.sum(h.unique_item())
    loss.backward()

    max_allocated = torch.cuda.max_memory_allocated() / 1024**2

    del h, loss

    assert (
        low_memory_max_allocated * 1.2 < max_allocated
        or identity_model.get_device() == "cpu"
    )
