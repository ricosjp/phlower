import logging

import numpy as np
import pytest
import torch
from phlower_tensor import PhlowerTensor, SimulationField, phlower_tensor
from phlower_tensor.collections import phlower_tensor_collection
from phlower_tensor.functionals import spmm

from phlower.nn import ConjugateGradientSolver
from phlower.nn._core_modules._conjugate_gradient_solver import SparseCGSolver
from phlower.utils import get_logger

_logger = get_logger(__name__)


def test__can_call_parameters():
    model = ConjugateGradientSolver(matrix_name="A")

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


def _eye(rank: int) -> torch.Tensor:
    indices = torch.stack([torch.arange(rank)] * 2, dim=0)
    values = torch.ones(rank)
    eye = torch.sparse_coo_tensor(indices, values, size=(rank, rank))
    return eye


def _generate_spd_mat(
    rank: int, dimension: dict[str, int] | None = None
) -> PhlowerTensor:
    tensor = _make_solvable(_generate_sparse_tensor(rank))

    return phlower_tensor(tensor, dimension=dimension)


def _generate_sparse_tensor(rank: int, nnz_per_raw: int = 10) -> torch.Tensor:
    indices = torch.randint(rank, (2, rank * nnz_per_raw))
    values = torch.rand(rank * nnz_per_raw)
    tensor = torch.sparse_coo_tensor(
        indices, values, size=(rank, rank)
    ).coalesce()
    return tensor


def _generate_laplacian(
    rank: int, dimension: dict[str, int] | None = None
) -> PhlowerTensor:
    main_diag = torch.stack([torch.arange(rank), torch.arange(rank)], dim=0)
    upper_diag = torch.stack(
        [torch.arange(rank - 1), torch.arange(1, rank)], dim=0
    )
    lower_diag = torch.stack(
        [torch.arange(1, rank), torch.arange(rank - 1)], dim=0
    )

    indices = torch.hstack((main_diag, upper_diag, lower_diag))

    diag = -torch.ones(rank) * 2.0
    diag[0] = -1.0
    diag[-1] = -1.0
    values = torch.cat(
        (
            diag,
            torch.ones(rank - 1),
            torch.ones(rank - 1),
        )
    )

    laplacian = torch.sparse_coo_tensor(
        indices, values, (rank, rank)
    ).coalesce()
    return phlower_tensor(laplacian, dimension=dimension)


def _generate_2d_laplacian() -> PhlowerTensor:
    np_mat = np.array(
        [
            # 0  1  2  3  4  5  6  7
            [2, -1, 0, 0, -1, 0, 0, 0],
            [-1, 3, -1, 0, 0, -1, 0, 0],
            [0, -1, 3, -1, 0, 0, -1, 0],
            [0, 0, -1, 2, 0, 0, 0, -1],
            [-1, 0, 0, 0, 2, -1, 0, 0],
            [0, -1, 0, 0, -1, 3, -1, 0],
            [0, 0, -1, 0, 0, -1, 3, -1],
            [0, 0, 0, -1, 0, 0, -1, 2],
        ]
    )
    np.testing.assert_almost_equal(np_mat - np_mat.T, 0)
    return phlower_tensor(
        torch.from_numpy(np_mat).to(torch.float32).to_sparse_coo()
    )


def _make_solvable(
    tensor: torch.Tensor, coeff_offdiag: float = 1.0
) -> torch.Tensor:
    tensor = tensor.transpose(0, 1) @ tensor
    tensor = (
        coeff_offdiag * tensor + _eye(tensor.shape[0]).to(tensor.device)
    ).coalesce()
    return tensor


@pytest.mark.parametrize("rank", [10, 100, 1000])
@pytest.mark.parametrize("component_shape", [(1,), (4,), (3, 4)])
@pytest.mark.parametrize("batch_solve", [True, False])
def test__can_solve_linear_system(
    rank: int, component_shape: tuple[int], batch_solve: bool
):
    b = phlower_tensor(torch.randn([rank] + list(component_shape)))
    spmat = _generate_spd_mat(rank)
    field = SimulationField(
        field_tensors=phlower_tensor_collection({"spmat": spmat})
    )

    cg = ConjugateGradientSolver(
        matrix_name="spmat", rtol=1e-8, batch_solve=batch_solve
    )
    solution = cg(phlower_tensor_collection({"b": b}), field_data=field)

    rmse = torch.sqrt(torch.mean((spmm(spmat, solution) - b) ** 2))
    assert rmse < 1e-3


@pytest.mark.gpu_test
@pytest.mark.parametrize("force_cpu", [True, False])
@pytest.mark.parametrize("batch_solve", [True, False])
def test__can_consider_dirichlet(
    force_cpu: bool, auto_device: str, batch_solve: bool
):
    rank = 60

    # Consider vector-valued problem like the structural analysis
    b = phlower_tensor(torch.ones((round(rank / 3), 3, 2))).to(
        device=auto_device
    )
    b[:, :, 1] = 0

    dirichlet = torch.ones((round(rank / 3), 3, 2)) * torch.nan
    dirichlet[0, 0, 0] = 0.0
    dirichlet[-1, -1, 0] = 0.0
    dirichlet[0, 0, 1] = -1.0
    dirichlet[-1, -1, 1] = 2.0

    dirichlet = phlower_tensor(dirichlet.to(device=auto_device))

    # Use negative Laplacian to solve L u + f = 0
    spmat = -_generate_laplacian(rank).to(device=auto_device)

    # Confirm that the matrix is not solvable without Dirichlet
    assert torch.linalg.matrix_rank(spmat.to_tensor().to_dense()) < rank

    field = SimulationField(
        field_tensors=phlower_tensor_collection({"spmat": spmat})
    )

    cg = ConjugateGradientSolver(
        matrix_name="spmat",
        dirichlet_name="dirichlet",
        rtol=0,
        atol=1e-10,
        batch_solve=batch_solve,
        force_cpu=force_cpu,
    )
    solution = cg(
        phlower_tensor_collection({"b": b, "dirichlet": dirichlet}),
        field_data=field,
    )

    x = np.arange(rank)
    ref_0 = -0.5 * x * (x - (rank - 1))
    ref_1 = np.linspace(-1.0, 2.0, rank)

    np.testing.assert_almost_equal(
        np.ravel(solution[..., 0].numpy()) / np.max(ref_0),
        ref_0 / np.max(ref_0),
        decimal=4,
    )
    np.testing.assert_almost_equal(
        np.ravel(solution[..., 1].numpy()) / np.max(ref_1),
        ref_1 / np.max(ref_1),
        decimal=4,
    )


@pytest.mark.gpu_test
@pytest.mark.parametrize("batch_solve", [True, False])
def test__can_satisfy_dirichlet(auto_device: str, batch_solve: bool):
    rank = 5

    # Consider vector-valued problem like the structural analysis
    b = phlower_tensor(torch.randn(rank, 2)).to(device=auto_device)

    dirichlet = torch.ones((rank, 2)) * torch.nan
    dirichlet[0:3, :] = torch.randn(3, 2)
    dirichlet[4:, :] = torch.randn(rank - 4, 2)
    dirichlet = phlower_tensor(dirichlet.to(device=auto_device))

    # Use negative Laplacian to solve L u + f = 0
    spmat = _generate_spd_mat(rank).to(device=auto_device)

    field = SimulationField(
        field_tensors=phlower_tensor_collection({"spmat": spmat})
    )
    cg = ConjugateGradientSolver(
        matrix_name="spmat",
        dirichlet_name="dirichlet",
        rtol=0,
        atol=1e-10,
        batch_solve=batch_solve,
    )
    solution = cg(
        phlower_tensor_collection({"b": b, "dirichlet": dirichlet}),
        field_data=field,
    )

    mask_dirichlet = ~torch.isnan(dirichlet.to_tensor())
    np.testing.assert_almost_equal(
        solution[mask_dirichlet].numpy(),
        dirichlet[mask_dirichlet].numpy(),
        decimal=5,
    )


def test__can_solve_2d_laplace():
    # 2D grid with 2x4 points
    spmat = _generate_2d_laplacian()
    n = spmat.shape[0]

    left_dirichlet_indices = torch.tensor([0, 4])
    right_dirichlet_indices = torch.tensor([3, 7])
    dirichlet = torch.ones(n) * torch.nan
    dirichlet[left_dirichlet_indices] = -1
    dirichlet[right_dirichlet_indices] = 1
    dirichlet = phlower_tensor(dirichlet)
    b = phlower_tensor(torch.zeros(n))

    field = SimulationField(
        field_tensors=phlower_tensor_collection({"spmat": spmat})
    )
    cg = ConjugateGradientSolver(
        matrix_name="spmat",
        dirichlet_name="dirichlet",
        rtol=0,
        atol=1e-10,
    )
    solution = cg(
        phlower_tensor_collection({"b": b, "dirichlet": dirichlet}),
        field_data=field,
    )
    expected = np.reshape(
        np.repeat(np.linspace(-1, 1, 4)[None, :], 2, axis=0), (-1, 1)
    )
    np.testing.assert_almost_equal(solution, expected)


@pytest.mark.gpu_test
@pytest.mark.parametrize("force_cpu", [True, False])
@pytest.mark.parametrize("batch_solve", [True, False])
def test__can_consider_x0(
    force_cpu: bool,
    auto_device: str,
    batch_solve: bool,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.WARNING)
    rank = 100
    component_shape = (16,)

    b = phlower_tensor(torch.randn([rank] + list(component_shape))).to(
        auto_device
    )
    spmat = _generate_spd_mat(rank).to(auto_device)
    field = SimulationField(
        field_tensors=phlower_tensor_collection({"spmat": spmat})
    )

    cg = ConjugateGradientSolver(
        matrix_name="spmat", rtol=1e-8, batch_solve=batch_solve
    )
    solution = cg(phlower_tensor_collection({"b": b}), field_data=field)

    # Confirm the solver not convergent with a smaller maxiter
    cg_short = ConjugateGradientSolver(
        matrix_name="spmat", rtol=1e-8, batch_solve=batch_solve, maxiter=1
    )
    cg_short(phlower_tensor_collection({"b": b}), field_data=field)
    assert "CG solver not converged" in caplog.text

    # Confirm the solver convergent even with a smaller maxiter, under x0
    cg_x0 = ConjugateGradientSolver(
        matrix_name="spmat",
        rtol=1e-8,
        batch_solve=batch_solve,
        maxiter=1,
        x0_name="x0",
    )
    solution_x0 = cg_x0(
        phlower_tensor_collection({"b": b, "x0": solution}), field_data=field
    )

    rmse = torch.sqrt(torch.mean((solution_x0 - solution) ** 2))
    assert rmse < 1e-4


def test__raises_time_series():
    rank = 10
    b = phlower_tensor(torch.randn((20, rank, 3, 2)), is_time_series=True)
    spmat = _generate_spd_mat(rank)
    field = SimulationField(
        field_tensors=phlower_tensor_collection({"spmat": spmat})
    )
    cg = ConjugateGradientSolver(matrix_name="spmat", rtol=1e-8)
    with pytest.raises(
        NotImplementedError, match="Time series is not supported"
    ):
        cg(phlower_tensor_collection({"b": b}), field_data=field)


@pytest.mark.parametrize(
    "mat_dimension, b_dimension",
    [
        (None, None),
        ({}, {}),
        ({}, {"L": 1, "T": -1}),
        ({"L": 1, "T": -1}, {}),
        ({"L": -2}, {"L": 1, "T": -1}),
    ],
)
@pytest.mark.parametrize("has_dirichlet", [True, False])
def test__dimension_consistent(
    mat_dimension: dict[str, int] | None,
    b_dimension: dict[str, int] | None,
    has_dirichlet: bool,
):
    rank = 10
    b = phlower_tensor(
        torch.randn((10, 3, 2)),
        dimension=b_dimension,
    )
    spmat = _generate_spd_mat(rank, dimension=mat_dimension)
    if has_dirichlet:
        if b_dimension is None:
            dirichlet_dimension = None
        else:
            dirichlet_dimension = b.dimension / spmat.dimension
        dirichlet_tensor = torch.ones_like(b.to_tensor()) * torch.nan
        dirichlet_tensor[0] = 0
        dirichlet_tensor[-1] = 1
        dirichlet = phlower_tensor(
            dirichlet_tensor,
            dimension=dirichlet_dimension,
        )
        tensors = phlower_tensor_collection({"b": b, "dirichlet": dirichlet})
        dirichlet_name = "dirichlet"
    else:
        tensors = phlower_tensor_collection({"b": b})
        dirichlet_name = None

    field = SimulationField(
        field_tensors=phlower_tensor_collection({"spmat": spmat})
    )

    cg = ConjugateGradientSolver(
        matrix_name="spmat", dirichlet_name=dirichlet_name, rtol=1e-8
    )
    solution = cg(tensors, field_data=field)
    if mat_dimension is None or b_dimension is None:
        assert solution.dimension is None
    else:
        assert solution.dimension * spmat.dimension == b.dimension


@pytest.mark.parametrize("rank", [10, 100])
@pytest.mark.parametrize("component_shape", [(1,), (4,)])
@pytest.mark.parametrize("batch_solve", [True, False])
def test__gradient_correct(
    rank: int, component_shape: tuple[int], batch_solve: bool
):
    b = torch.randn([rank] + list(component_shape)).requires_grad_(True)

    # requires_grad == False because no support for sparse grad comparison
    m = _generate_spd_mat(rank).to_tensor()

    def func(spmat: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return SparseCGSolver.apply(
            spmat, b, None, 0.0, 1e-15, None, batch_solve
        )

    assert torch.autograd.gradcheck(
        func, (m.to(torch.float64), b.to(torch.float64)), eps=1e-6, atol=1e-5
    )


@pytest.mark.parametrize("rank", [10, 100])
@pytest.mark.parametrize("component_shape", [(1,), (4,)])
@pytest.mark.parametrize("batch_solve", [True, False])
def test__gradient_consistent_with_dense_solve(
    rank: int, component_shape: tuple[int], batch_solve: bool
):

    seed_b = torch.randn([rank] + list(component_shape), dtype=torch.float64)
    b_for_sparse = seed_b.clone().requires_grad_(True)
    # b_ = b.clone()
    b_for_dense = seed_b.clone().requires_grad_(True)
    seed_m = _generate_spd_mat(rank).to_tensor().to(torch.float64)

    m_for_sparse = (
        torch.sparse_coo_tensor(
            indices=seed_m.indices(),
            values=seed_m.values(),
            size=seed_m.size(),
        )
        .coalesce()
        .requires_grad_(True)
    )
    m_for_dense = (
        torch.sparse_coo_tensor(
            indices=seed_m.indices(),
            values=seed_m.values(),
            size=seed_m.size(),
        )
        .coalesce()
        .to_dense()
        .requires_grad_(True)
    )

    l_sparse = torch.sum(
        SparseCGSolver.apply(
            m_for_sparse, b_for_sparse, None, 0.0, 1e-15, None, batch_solve
        )
    )
    l_sparse.backward()

    l_dense = torch.sum(torch.linalg.solve(m_for_dense, b_for_dense))
    l_dense.backward()
    assert (
        torch.mean((b_for_sparse.grad - b_for_dense.grad) ** 2) ** 0.5
        / torch.max(torch.abs(seed_b))
        < 5.0e-2
    )
    assert (
        torch.mean((m_for_sparse.grad.to_dense() - m_for_dense.grad) ** 2)
        ** 0.5
        / torch.max(torch.abs(seed_m.values()))
        < 5.0e-2
    )


@pytest.mark.gpu_test
@pytest.mark.parametrize("has_dirichlet", [True, False])
def test__can_backward_wrt_b(has_dirichlet: bool, auto_device: str):
    rank = 20
    n_epoch = 20

    gt_b = phlower_tensor(
        torch.stack(
            [
                torch.linspace(0.0, 2.0, rank),
                torch.linspace(1.0, 2.0, rank),
                torch.linspace(0.0, -1.0, rank),
            ],
            dim=-1,
        ),
    ).to(auto_device)
    spmat = _generate_spd_mat(rank).to(auto_device)
    field = SimulationField(
        field_tensors=phlower_tensor_collection({"spmat": spmat})
    )

    if has_dirichlet:
        dirichlet_tensor = torch.ones_like(gt_b.to_tensor()) * torch.nan
        dirichlet_tensor[0] = 0
        dirichlet_tensor[-1] = 0
        dirichlet = phlower_tensor(
            dirichlet_tensor,
        ).to(auto_device)
        dict_dirichlet = {"dirichlet": dirichlet}
        dirichlet_name = "dirichlet"
    else:
        dict_dirichlet = {}
        dirichlet_name = None

    cg_for_gt = ConjugateGradientSolver(
        matrix_name="spmat", dirichlet_name=dirichlet_name
    )
    dict_gt = {"b": gt_b}
    dict_gt.update(dict_dirichlet)
    gt_x = cg_for_gt(phlower_tensor_collection(dict_gt), field_data=field)

    cg = ConjugateGradientSolver(
        matrix_name="spmat", x0_name="x0", dirichlet_name=dirichlet_name
    )

    t_b = torch.randn(gt_b.shape, device=auto_device, requires_grad=True)
    b = phlower_tensor(t_b)
    initial_rmse_b = torch.sqrt(torch.mean((b - gt_b)[1:-1] ** 2))

    optimizer = torch.optim.Adam([b.to_tensor()], lr=0.1)
    for _ in range(n_epoch):
        optimizer.zero_grad()
        dict_input = {"b": b, "x0": gt_x * 0.9}
        dict_input.update(dict_dirichlet)
        x = cg(phlower_tensor_collection(dict_input), field_data=field)

        loss = torch.mean((gt_x - x) ** 2)
        loss.backward()
        optimizer.step()

    assert torch.sqrt(torch.mean((x - gt_x) ** 2)) < 0.5
    if has_dirichlet:
        np.testing.assert_almost_equal(x[0].numpy(), 0)
        np.testing.assert_almost_equal(x[-1].numpy(), 0)
        assert torch.sqrt(torch.mean((b - gt_b)[1:-1] ** 2)) < initial_rmse_b
    else:
        assert torch.sqrt(torch.mean((b - gt_b) ** 2)) < initial_rmse_b


@pytest.mark.gpu_test
@pytest.mark.parametrize("has_dirichlet", [True, False])
def test__can_backward_wrt_mat(has_dirichlet: bool, auto_device: str):
    rank = 10
    n_epoch = 50
    coeff_offdiag = 1.0e-3

    b = phlower_tensor(
        torch.stack(
            [
                torch.linspace(0.0, 2.0, rank),
                torch.linspace(1.0, 2.0, rank),
                torch.linspace(0.0, -1.0, rank),
            ],
            dim=-1,
        ),
    ).to(auto_device)
    raw_sparse = _generate_sparse_tensor(rank, nnz_per_raw=1).to(auto_device)
    gt_spmat = phlower_tensor(_make_solvable(raw_sparse, coeff_offdiag))
    field = SimulationField(
        field_tensors=phlower_tensor_collection({"spmat": gt_spmat})
    )

    if has_dirichlet:
        dirichlet_tensor = torch.ones_like(b.to_tensor()) * torch.nan
        dirichlet_tensor[0] = 0
        dirichlet_tensor[-1] = 1.0
        dirichlet = phlower_tensor(
            dirichlet_tensor,
        )
        dict_dirichlet = {"dirichlet": dirichlet}
        dirichlet_name = "dirichlet"
    else:
        dict_dirichlet = {}
        dirichlet_name = None

    cg_for_gt = ConjugateGradientSolver(
        matrix_name="spmat",
        dirichlet_name=dirichlet_name,
    )
    dict_gt = {"b": b}
    dict_gt.update(dict_dirichlet)
    gt_x = cg_for_gt(phlower_tensor_collection(dict_gt), field_data=field)

    cg = ConjugateGradientSolver(
        matrix_name="spmat",
        x0_name="x0",
        dirichlet_name=dirichlet_name,
    )
    dict_input = {"b": b, "x0": gt_x * 0.9}
    dict_input.update(dict_dirichlet)

    values = torch.randn(
        raw_sparse.values().shape, device=auto_device, requires_grad=True
    )
    raw_mat = torch.sparse_coo_tensor(
        indices=raw_sparse.indices(),
        values=values,
        size=raw_sparse.size(),
    )
    spmat = phlower_tensor(_make_solvable(raw_mat, coeff_offdiag))

    scale = torch.max(gt_spmat.values())
    initial_rmse_mat = (
        torch.sqrt(
            torch.sum((gt_spmat.values() - spmat.values()) ** 2) / rank / rank
        )
        / scale
    )

    optimizer = torch.optim.Adam([values], lr=0.001)
    for _ in range(n_epoch):
        optimizer.zero_grad()

        raw_mat = torch.sparse_coo_tensor(
            indices=raw_sparse.indices(),
            values=values,
            size=raw_sparse.size(),
        )
        spmat = phlower_tensor(_make_solvable(raw_mat, coeff_offdiag))

        x = cg(
            phlower_tensor_collection(dict_input),
            field_data=SimulationField(
                field_tensors=phlower_tensor_collection({"spmat": spmat})
            ),
        )

        loss = torch.mean((gt_x - x) ** 2)
        loss.backward()
        optimizer.step()

    raw_mat = torch.sparse_coo_tensor(
        indices=raw_sparse.indices(),
        values=values,
        size=raw_sparse.size(),
    )
    spmat = phlower_tensor(_make_solvable(raw_mat, coeff_offdiag))
    values = spmat.values()

    assert torch.sqrt(torch.mean((x - gt_x) ** 2)) < 0.5
    assert (
        torch.sqrt(
            torch.sum((gt_spmat.values() - spmat.values()) ** 2) / rank / rank
        )
        / scale
        < initial_rmse_mat
    )
    if has_dirichlet:
        np.testing.assert_almost_equal(x[0].numpy(), 0, decimal=5)
        np.testing.assert_almost_equal(x[-1].numpy(), 1, decimal=5)
