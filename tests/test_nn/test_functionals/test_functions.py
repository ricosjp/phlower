import numpy as np
import phlower
import pytest
import torch
from phlower import phlower_tensor
from phlower._base import phlower_array
from phlower.utils.exceptions import PhlowerIncompatibleTensorError
from scipy import sparse as sp
from scipy.stats import ortho_group


@pytest.mark.parametrize(
    "size, is_time_series, repeat",
    [
        ((10, 1), False, 1),
        ((10, 16), False, 1),
        ((10, 3, 16), False, 1),
        ((4, 10, 1), True, 1),
        ((4, 10, 16), True, 1),
        ((4, 10, 3, 16), True, 1),
        ((10, 1), False, 5),
        ((10, 16), False, 5),
        ((10, 3, 16), False, 5),
        ((4, 10, 1), True, 5),
        ((4, 10, 16), True, 5),
        ((4, 10, 3, 16), True, 5),
    ],
)
def test__spmm(size: tuple[int], is_time_series: bool, repeat: bool):
    _tensor = phlower_tensor(torch.rand(*size), is_time_series=is_time_series)
    n = _tensor.n_vertices()
    sparse = phlower_tensor(torch.rand(n, n).to_sparse())

    actual_spmm = phlower.nn.functional.spmm(sparse, _tensor, repeat=repeat)
    np_actual_spmm = actual_spmm.to_numpy()
    sp_sparse = sp.coo_array(sparse.to_tensor().to_dense().numpy())
    np_dense = _tensor.to_tensor().numpy()

    def assert_correct(actual: np.ndarray, array: np.ndarray):
        dim_feat = len(array.shape) - 1
        if dim_feat == 1:
            desired = array
            for _ in range(repeat):
                desired = sp_sparse @ desired
            norm = np.mean(np.linalg.norm(desired, axis=-1))
            np.testing.assert_almost_equal(
                actual / norm, desired / norm, decimal=5
            )
            return

        for i in range(array.shape[1]):
            assert_correct(actual[:, i], array[:, i])
        return

    if is_time_series:
        assert actual_spmm.is_time_series
        for t in range(size[0]):
            assert_correct(np_actual_spmm[t], np_dense[t])
    else:
        assert_correct(np_actual_spmm, np_dense)


@pytest.mark.parametrize(
    "size, sparse_size, is_time_series, desired_shape",
    [
        ((5, 10), (8, 5), False, (8, 10)),
        ((4, 10, 16), (5, 10), True, (4, 5, 16)),
    ],
)
def test__spmm_with_not_squared_sparse_matrix(
    size: tuple[int],
    sparse_size: tuple[int],
    is_time_series: bool,
    desired_shape: tuple[int],
):
    _tensor = phlower_tensor(torch.rand(*size), is_time_series=is_time_series)

    sparse_array = phlower_array(
        sp.random(*sparse_size, density=0.4, dtype=np.float32)
    )
    sparse_tensor = phlower_tensor(sparse_array.to_tensor())

    actual_spmm = phlower.nn.functional.spmm(sparse_tensor, _tensor, repeat=1)

    assert actual_spmm.shape == desired_shape


@pytest.mark.parametrize(
    "size, is_time_series, is_voxel, desired_pattern",
    [
        ((10, 1), False, False, "n...f,n...f->nf"),
        ((10, 16), False, False, "n...f,n...f->nf"),
        ((10, 3, 16), False, False, "n...f,n...f->nf"),
        ((10, 3, 3, 16), False, False, "n...f,n...f->nf"),
        ((4, 10, 1), True, False, "tn...f,tn...f->tnf"),
        ((4, 10, 16), True, False, "tn...f,tn...f->tnf"),
        ((4, 10, 3, 16), True, False, "tn...f,tn...f->tnf"),
        ((4, 10, 3, 3, 16), True, False, "tn...f,tn...f->tnf"),
        ((10, 10, 10, 1), False, True, "xyz...f,xyz...f->xyzf"),
        ((10, 10, 10, 16), False, True, "xyz...f,xyz...f->xyzf"),
        ((10, 10, 10, 3, 16), False, True, "xyz...f,xyz...f->xyzf"),
        ((10, 10, 10, 3, 3, 16), False, True, "xyz...f,xyz...f->xyzf"),
        ((4, 10, 10, 10, 1), True, True, "txyz...f,txyz...f->txyzf"),
        ((4, 10, 10, 10, 16), True, True, "txyz...f,txyz...f->txyzf"),
        ((4, 10, 10, 10, 3, 16), True, True, "txyz...f,txyz...f->txyzf"),
        ((4, 10, 10, 10, 3, 3, 16), True, True, "txyz...f,txyz...f->txyzf"),
    ],
)
@pytest.mark.parametrize(
    "dimension",
    [
        None,
        [[-1], [2], [0], [0], [0], [0], [0]],
        [[1], [0], [1], [0], [0], [0], [0]],
        [[-1], [-1], [2], [0], [1], [0], [0]],
    ],
)
def test_contraction_one_argument(
    size: tuple[int],
    is_time_series: bool,
    is_voxel: bool,
    desired_pattern: str,
    dimension: list[list[int]],
):
    torch_tensor = torch.rand(*size)
    x = phlower_tensor(
        torch_tensor,
        dimension=dimension,
        is_time_series=is_time_series,
        is_voxel=is_voxel,
    )
    actual = phlower.nn.functional.contraction(x)
    desired = torch.einsum(desired_pattern, torch_tensor, torch_tensor).numpy()
    np.testing.assert_almost_equal(actual.to_tensor().numpy(), desired)

    assert actual.is_time_series == is_time_series
    assert actual.is_voxel == is_voxel
    assert actual.rank() == 0

    if dimension is not None:
        actual_dimension = actual.dimension._tensor.numpy()
        desired_dimension = np.array(dimension) * 2
        np.testing.assert_almost_equal(actual_dimension, desired_dimension)
    else:
        assert actual.dimension is None


@pytest.mark.parametrize(
    "size_x, size_y, x_is_time_series, y_is_time_series, is_voxel, "
    "desired_pattern, desired_rank",
    [
        # Base
        ((10, 16), (10, 16), False, False, False, "nf,nf->nf", 0),
        ((10, 3, 16), (10, 16), False, False, False, "npf,nf->npf", 1),
        ((10, 16), (10, 3, 16), False, False, False, "nf,npf->npf", 1),
        ((10, 3, 16), (10, 3, 16), False, False, False, "npf,npf->nf", 0),
        ((10, 3, 3, 16), (10, 16), False, False, False, "npqf,nf->npqf", 2),
        ((10, 3, 3, 16), (10, 3, 16), False, False, False, "npqf,npf->nqf", 1),
        (
            (10, 3, 3, 16),
            (10, 3, 3, 16),
            False,
            False,
            False,
            "npqf,npqf->nf",
            0,
        ),
        # X time series
        ((4, 10, 16), (10, 16), True, False, False, "tnf,nf->tnf", 0),
        ((4, 10, 3, 16), (10, 16), True, False, False, "tnpf,nf->tnpf", 1),
        ((4, 10, 16), (10, 3, 16), True, False, False, "tnf,npf->tnpf", 1),
        ((4, 10, 3, 16), (10, 3, 16), True, False, False, "tnpf,npf->tnf", 0),
        ((4, 10, 3, 3, 16), (10, 16), True, False, False, "tnpqf,nf->tnpqf", 2),
        (
            (4, 10, 3, 3, 16),
            (10, 3, 16),
            True,
            False,
            False,
            "tnpqf,npf->tnqf",
            1,
        ),
        (
            (4, 10, 3, 3, 16),
            (10, 3, 3, 16),
            True,
            False,
            False,
            "tnpqf,npqf->tnf",
            0,
        ),
        # Y time series
        ((10, 16), (4, 10, 16), False, True, False, "nf,tnf->tnf", 0),
        ((10, 3, 16), (4, 10, 16), False, True, False, "npf,tnf->tnpf", 1),
        ((10, 16), (4, 10, 3, 16), False, True, False, "nf,tnpf->tnpf", 1),
        ((10, 3, 16), (4, 10, 3, 16), False, True, False, "npf,tnpf->tnf", 0),
        ((10, 3, 3, 16), (4, 10, 16), False, True, False, "npqf,tnf->tnpqf", 2),
        (
            (10, 3, 3, 16),
            (4, 10, 3, 16),
            False,
            True,
            False,
            "npqf,tnpf->tnqf",
            1,
        ),
        (
            (10, 3, 3, 16),
            (4, 10, 3, 3, 16),
            False,
            True,
            False,
            "npqf,tnpqf->tnf",
            0,
        ),
        # X Y time series
        ((4, 10, 16), (4, 10, 16), True, True, False, "tnf,tnf->tnf", 0),
        ((4, 10, 3, 16), (4, 10, 16), True, True, False, "tnpf,tnf->tnpf", 1),
        ((4, 10, 16), (4, 10, 3, 16), True, True, False, "tnf,tnpf->tnpf", 1),
        (
            (4, 10, 3, 16),
            (4, 10, 3, 16),
            True,
            True,
            False,
            "tnpf,tnpf->tnf",
            0,
        ),
        (
            (4, 10, 3, 3, 16),
            (4, 10, 16),
            True,
            True,
            False,
            "tnpqf,tnf->tnpqf",
            2,
        ),
        (
            (4, 10, 3, 3, 16),
            (4, 10, 3, 16),
            True,
            True,
            False,
            "tnpqf,tnpf->tnqf",
            1,
        ),
        (
            (4, 10, 3, 3, 16),
            (4, 10, 3, 3, 16),
            True,
            True,
            False,
            "tnpqf,tnpqf->tnf",
            0,
        ),
        # X time series, X Y voxel
        (
            (4, 10, 10, 10, 16),
            (10, 10, 10, 16),
            True,
            False,
            True,
            "txyzf,xyzf->txyzf",
            0,
        ),
        (
            (4, 10, 10, 10, 3, 16),
            (10, 10, 10, 16),
            True,
            False,
            True,
            "txyzpf,xyzf->txyzpf",
            1,
        ),
        (
            (4, 10, 10, 10, 16),
            (10, 10, 10, 3, 16),
            True,
            False,
            True,
            "txyzf,xyzpf->txyzpf",
            1,
        ),
        (
            (4, 10, 10, 10, 3, 16),
            (10, 10, 10, 3, 16),
            True,
            False,
            True,
            "txyzpf,xyzpf->txyzf",
            0,
        ),
        (
            (4, 10, 10, 10, 3, 3, 16),
            (10, 10, 10, 16),
            True,
            False,
            True,
            "txyzpqf,xyzf->txyzpqf",
            2,
        ),
        (
            (4, 10, 10, 10, 3, 3, 16),
            (10, 10, 10, 3, 16),
            True,
            False,
            True,
            "txyzpqf,xyzpf->txyzqf",
            1,
        ),
        (
            (4, 10, 10, 10, 3, 3, 16),
            (10, 10, 10, 3, 3, 16),
            True,
            False,
            True,
            "txyzpqf,xyzpqf->txyzf",
            0,
        ),
        # X Y time series, X Y voxel
        (
            (4, 10, 10, 10, 16),
            (4, 10, 10, 10, 16),
            True,
            True,
            True,
            "txyzf,txyzf->txyzf",
            0,
        ),
        (
            (4, 10, 10, 10, 3, 16),
            (4, 10, 10, 10, 16),
            True,
            True,
            True,
            "txyzpf,txyzf->txyzpf",
            1,
        ),
        (
            (4, 10, 10, 10, 16),
            (4, 10, 10, 10, 3, 16),
            True,
            True,
            True,
            "txyzf,txyzpf->txyzpf",
            1,
        ),
        (
            (4, 10, 10, 10, 3, 16),
            (4, 10, 10, 10, 3, 16),
            True,
            True,
            True,
            "txyzpf,txyzpf->txyzf",
            0,
        ),
        (
            (4, 10, 10, 10, 3, 3, 16),
            (4, 10, 10, 10, 16),
            True,
            True,
            True,
            "txyzpqf,txyzf->txyzpqf",
            2,
        ),
        (
            (4, 10, 10, 10, 3, 3, 16),
            (4, 10, 10, 10, 3, 16),
            True,
            True,
            True,
            "txyzpqf,txyzpf->txyzqf",
            1,
        ),
        (
            (4, 10, 10, 10, 3, 3, 16),
            (4, 10, 10, 10, 3, 3, 16),
            True,
            True,
            True,
            "txyzpqf,txyzpqf->txyzf",
            0,
        ),
    ],
)
@pytest.mark.parametrize(
    "dimension_x",
    [
        None,
        [[-1], [2], [0], [0], [0], [0], [0]],
        [[1], [0], [1], [0], [0], [0], [0]],
        [[-1], [-1], [2], [0], [1], [0], [0]],
    ],
)
@pytest.mark.parametrize(
    "dimension_y",
    [
        None,
        [[-1], [2], [0], [0], [0], [0], [0]],
        [[1], [0], [1], [0], [0], [0], [0]],
        [[-1], [-1], [2], [0], [1], [0], [0]],
    ],
)
def test_contraction_two_arguments(
    size_x: tuple[int],
    size_y: tuple[int],
    x_is_time_series: bool,
    y_is_time_series: bool,
    is_voxel: bool,
    desired_pattern: str,
    desired_rank: int,
    dimension_x: list[list[int]] | None,
    dimension_y: list[list[int]] | None,
):
    t_x = torch.rand(*size_x)
    x = phlower_tensor(
        t_x,
        dimension=dimension_x,
        is_time_series=x_is_time_series,
        is_voxel=is_voxel,
    )

    t_y = torch.rand(*size_y)
    y = phlower_tensor(
        t_y,
        dimension=dimension_y,
        is_time_series=y_is_time_series,
        is_voxel=is_voxel,
    )

    actual = phlower.nn.functional.contraction(x, y)
    desired = torch.einsum(desired_pattern, t_x, t_y).numpy()
    np.testing.assert_almost_equal(actual.to_tensor().numpy(), desired)

    assert actual.is_time_series == x_is_time_series or y_is_time_series
    assert actual.is_voxel == is_voxel
    assert actual.rank() == desired_rank

    if dimension_x is not None and dimension_y is not None:
        actual_dimension = actual.dimension._tensor.numpy()
        desired_dimension = np.array(dimension_x) + np.array(dimension_y)
        np.testing.assert_almost_equal(actual_dimension, desired_dimension)
    else:
        assert actual.dimension is None


def test_contraction_raises_phlower_incompatible_tensor_error():
    x = phlower_tensor(torch.rand(10, 10, 10, 3, 16), is_voxel=True)
    y = phlower_tensor(torch.rand(10 * 10 * 10, 3, 16), is_voxel=False)
    with pytest.raises(PhlowerIncompatibleTensorError):
        phlower.nn.functional.contraction(x, y)


@pytest.mark.parametrize(
    "size_x, size_y, x_is_time_series, y_is_time_series, is_voxel, "
    "desired_pattern",
    [
        # Base
        ((10, 16), (10, 16), False, False, False, "nf,nf->nf"),
        ((10, 3, 16), (10, 16), False, False, False, "npf,nf->npf"),
        ((10, 16), (10, 3, 16), False, False, False, "nf,npf->npf"),
        ((10, 3, 16), (10, 3, 16), False, False, False, "npf,nqf->npqf"),
        ((10, 3, 3, 16), (10, 16), False, False, False, "npqf,nf->npqf"),
        ((10, 3, 3, 16), (10, 3, 16), False, False, False, "npqf,nrf->npqrf"),
        (
            (10, 3, 3, 16),
            (10, 3, 3, 16),
            False,
            False,
            False,
            "npqf,nrsf->npqrsf",
        ),
        # X time series
        ((4, 10, 16), (10, 16), True, False, False, "tnf,nf->tnf"),
        ((4, 10, 3, 16), (10, 16), True, False, False, "tnpf,nf->tnpf"),
        ((4, 10, 16), (10, 3, 16), True, False, False, "tnf,npf->tnpf"),
        ((4, 10, 3, 16), (10, 3, 16), True, False, False, "tnpf,nqf->tnpqf"),
        ((4, 10, 3, 3, 16), (10, 16), True, False, False, "tnpqf,nf->tnpqf"),
        (
            (4, 10, 3, 3, 16),
            (10, 3, 16),
            True,
            False,
            False,
            "tnpqf,nrf->tnpqrf",
        ),
        (
            (4, 10, 3, 3, 16),
            (10, 3, 3, 16),
            True,
            False,
            False,
            "tnpqf,nrsf->tnpqrsf",
        ),
        # Y time series
        ((10, 16), (4, 10, 16), False, True, False, "nf,tnf->tnf"),
        ((10, 3, 16), (4, 10, 16), False, True, False, "npf,tnf->tnpf"),
        ((10, 16), (4, 10, 3, 16), False, True, False, "nf,tnpf->tnpf"),
        ((10, 3, 16), (4, 10, 3, 16), False, True, False, "npf,tnqf->tnpqf"),
        ((10, 3, 3, 16), (4, 10, 16), False, True, False, "npqf,tnf->tnpqf"),
        (
            (10, 3, 3, 16),
            (4, 10, 3, 16),
            False,
            True,
            False,
            "npqf,tnrf->tnpqrf",
        ),
        (
            (10, 3, 3, 16),
            (4, 10, 3, 3, 16),
            False,
            True,
            False,
            "npqf,tnrsf->tnpqrsf",
        ),
        # X Y time series
        ((4, 10, 16), (4, 10, 16), True, True, False, "tnf,tnf->tnf"),
        ((4, 10, 3, 16), (4, 10, 16), True, True, False, "tnpf,tnf->tnpf"),
        ((4, 10, 16), (4, 10, 3, 16), True, True, False, "tnf,tnpf->tnpf"),
        ((4, 10, 3, 16), (4, 10, 3, 16), True, True, False, "tnpf,tnqf->tnpqf"),
        ((4, 10, 3, 3, 16), (4, 10, 16), True, True, False, "tnpqf,tnf->tnpqf"),
        (
            (4, 10, 3, 3, 16),
            (4, 10, 3, 16),
            True,
            True,
            False,
            "tnpqf,tnrf->tnpqrf",
        ),
        (
            (4, 10, 3, 3, 16),
            (4, 10, 3, 3, 16),
            True,
            True,
            False,
            "tnpqf,tnrsf->tnpqrsf",
        ),
        # X time series, X Y voxel
        (
            (4, 10, 10, 10, 16),
            (10, 10, 10, 16),
            True,
            False,
            True,
            "txyzf,xyzf->txyzf",
        ),
        (
            (4, 10, 10, 10, 3, 16),
            (10, 10, 10, 16),
            True,
            False,
            True,
            "txyzpf,xyzf->txyzpf",
        ),
        (
            (4, 10, 10, 10, 16),
            (10, 10, 10, 3, 16),
            True,
            False,
            True,
            "txyzf,xyzpf->txyzpf",
        ),
        (
            (4, 10, 10, 10, 3, 16),
            (10, 10, 10, 3, 16),
            True,
            False,
            True,
            "txyzpf,xyzqf->txyzpqf",
        ),
        (
            (4, 10, 10, 10, 3, 3, 16),
            (10, 10, 10, 16),
            True,
            False,
            True,
            "txyzpqf,xyzf->txyzpqf",
        ),
        (
            (4, 10, 10, 10, 3, 3, 16),
            (10, 10, 10, 3, 16),
            True,
            False,
            True,
            "txyzpqf,xyzrf->txyzpqrf",
        ),
        (
            (4, 10, 10, 10, 3, 3, 16),
            (10, 10, 10, 3, 3, 16),
            True,
            False,
            True,
            "txyzpqf,xyzrsf->txyzpqrsf",
        ),
        # X Y time series, X Y voxel
        (
            (4, 10, 10, 10, 16),
            (4, 10, 10, 10, 16),
            True,
            True,
            True,
            "txyzf,txyzf->txyzf",
        ),
        (
            (4, 10, 10, 10, 3, 16),
            (4, 10, 10, 10, 16),
            True,
            True,
            True,
            "txyzpf,txyzf->txyzpf",
        ),
        (
            (4, 10, 10, 10, 16),
            (4, 10, 10, 10, 3, 16),
            True,
            True,
            True,
            "txyzf,txyzpf->txyzpf",
        ),
        (
            (4, 10, 10, 10, 3, 16),
            (4, 10, 10, 10, 3, 16),
            True,
            True,
            True,
            "txyzpf,txyzqf->txyzpqf",
        ),
        (
            (4, 10, 10, 10, 3, 3, 16),
            (4, 10, 10, 10, 16),
            True,
            True,
            True,
            "txyzpqf,txyzf->txyzpqf",
        ),
        (
            (4, 10, 10, 10, 3, 3, 16),
            (4, 10, 10, 10, 3, 16),
            True,
            True,
            True,
            "txyzpqf,txyzrf->txyzpqrf",
        ),
        (
            (4, 10, 10, 10, 3, 3, 16),
            (4, 10, 10, 10, 3, 3, 16),
            True,
            True,
            True,
            "txyzpqf,txyzrsf->txyzpqrsf",
        ),
    ],
)
@pytest.mark.parametrize(
    "dimension_x",
    [
        None,
        [[-1], [2], [0], [0], [0], [0], [0]],
        [[1], [0], [1], [0], [0], [0], [0]],
        [[-1], [-1], [2], [0], [1], [0], [0]],
    ],
)
@pytest.mark.parametrize(
    "dimension_y",
    [
        None,
        [[-1], [2], [0], [0], [0], [0], [0]],
        [[1], [0], [1], [0], [0], [0], [0]],
        [[-1], [-1], [2], [0], [1], [0], [0]],
    ],
)
def test_tensor_product(
    size_x: tuple[int],
    size_y: tuple[int],
    x_is_time_series: bool,
    y_is_time_series: bool,
    is_voxel: bool,
    desired_pattern: str,
    dimension_x: list[list[int]] | None,
    dimension_y: list[list[int]] | None,
):
    t_x = torch.rand(*size_x)
    x = phlower_tensor(
        t_x,
        dimension=dimension_x,
        is_time_series=x_is_time_series,
        is_voxel=is_voxel,
    )

    t_y = torch.rand(*size_y)
    y = phlower_tensor(
        t_y,
        dimension=dimension_y,
        is_time_series=y_is_time_series,
        is_voxel=is_voxel,
    )

    actual = phlower.nn.functional.tensor_product(x, y)
    desired = torch.einsum(desired_pattern, t_x, t_y).numpy()
    np.testing.assert_almost_equal(actual.to_numpy(), desired)

    assert actual.is_time_series == x_is_time_series or y_is_time_series
    assert actual.is_voxel == is_voxel
    assert actual.rank() == x.rank() + y.rank()

    if dimension_x is not None and dimension_y is not None:
        actual_dimension = actual.dimension._tensor.numpy()
        desired_dimension = np.array(dimension_x) + np.array(dimension_y)
        np.testing.assert_almost_equal(actual_dimension, desired_dimension)
    else:
        assert actual.dimension is None


@pytest.mark.parametrize(
    "size, is_time_series, is_voxel, desired_pattern",
    [
        ((10, 1), False, False, None),
        ((10, 16), False, False, None),
        ((10, 3, 16), False, False, "pq,nqf->npf"),
        ((10, 3, 3, 16), False, False, "pq,rs,nqsf->nprf"),
        ((4, 10, 1), True, False, None),
        ((4, 10, 16), True, False, None),
        ((4, 10, 3, 16), True, False, "pq,tnqf->tnpf"),
        ((4, 10, 3, 3, 16), True, False, "pq,rs,tnqsf->tnprf"),
        ((10, 10, 10, 1), False, True, None),
        ((10, 10, 10, 16), False, True, None),
        ((10, 10, 10, 3, 16), False, True, "pq,xyzqf->xyzpf"),
        ((10, 10, 10, 3, 3, 16), False, True, "pq,rs,xyzqsf->xyzprf"),
        ((4, 10, 10, 10, 1), True, True, None),
        ((4, 10, 10, 10, 16), True, True, None),
        ((4, 10, 10, 10, 3, 16), True, True, "pq,txyzqf->txyzpf"),
        ((4, 10, 10, 10, 3, 3, 16), True, True, "pq,rs,txyzqsf->txyzprf"),
    ],
)
@pytest.mark.parametrize(
    "dimension",
    [
        None,
        [[-1], [2], [0], [0], [0], [0], [0]],
        [[1], [0], [1], [0], [0], [0], [0]],
        [[-1], [-1], [2], [0], [1], [0], [0]],
    ],
)
def test_apply_orthogonal_group(
    size: tuple[int],
    is_time_series: bool,
    is_voxel: bool,
    desired_pattern: str,
    dimension: list[list[int]] | None,
):
    orthogonal_matrix = torch.from_numpy(ortho_group.rvs(3).astype(np.float32))
    orthogonal_tensor = phlower_tensor(orthogonal_matrix)

    torch_tensor = torch.rand(*size)
    x = phlower_tensor(
        torch_tensor,
        dimension=dimension,
        is_time_series=is_time_series,
        is_voxel=is_voxel,
    )
    actual = phlower.nn.functional.apply_orthogonal_group(orthogonal_tensor, x)

    if desired_pattern is None:
        desired = torch_tensor.numpy()
    else:
        inputs = [orthogonal_matrix] * x.rank() + [torch_tensor]
        desired = torch.einsum(desired_pattern, *inputs).numpy()
    np.testing.assert_almost_equal(actual.to_numpy(), desired)

    assert actual.is_time_series == is_time_series
    assert actual.is_voxel == is_voxel
    assert actual.rank() == x.rank()

    if dimension is not None:
        actual_dimension = actual.dimension._tensor.numpy()
        desired_dimension = dimension
        np.testing.assert_almost_equal(actual_dimension, desired_dimension)
    else:
        assert actual.dimension is None


@pytest.mark.parametrize(
    "size, is_time_series, is_voxel, mean_dims",
    [
        ((10, 1), False, False, [0]),
        ((10, 16), False, False, [0]),
        ((10, 3, 16), False, False, [0]),
        ((10, 3, 3, 16), False, False, [0]),
        ((4, 10, 1), True, False, [1]),
        ((4, 10, 16), True, False, [1]),
        ((4, 10, 3, 16), True, False, [1]),
        ((4, 10, 3, 3, 16), True, False, [1]),
        ((10, 10, 10, 1), False, True, [0, 1, 2]),
        ((10, 10, 10, 16), False, True, [0, 1, 2]),
        ((10, 10, 10, 3, 16), False, True, [0, 1, 2]),
        ((10, 10, 10, 3, 3, 16), False, True, [0, 1, 2]),
        ((4, 10, 10, 10, 1), True, True, [1, 2, 3]),
        ((4, 10, 10, 10, 16), True, True, [1, 2, 3]),
        ((4, 10, 10, 10, 3, 16), True, True, [1, 2, 3]),
        ((4, 10, 10, 10, 3, 3, 16), True, True, [1, 2, 3]),
    ],
)
@pytest.mark.parametrize(
    "dimension",
    [
        None,
        [[-1], [2], [0], [0], [0], [0], [0]],
        [[1], [0], [1], [0], [0], [0], [0]],
        [[-1], [-1], [2], [0], [1], [0], [0]],
    ],
)
def test_spatial_mean_wo_weight(
    size: tuple[int],
    is_time_series: bool,
    is_voxel: bool,
    mean_dims: list[int],
    dimension: list[list[int]] | None,
):
    torch_tensor = torch.rand(*size)
    x = phlower_tensor(
        torch_tensor,
        dimension=dimension,
        is_time_series=is_time_series,
        is_voxel=is_voxel,
    )
    actual = phlower.nn.functional.spatial_mean(x)

    desired = torch_tensor
    for dim in mean_dims:
        desired = torch.mean(desired, dim=dim, keepdim=True)
    np.testing.assert_almost_equal(
        actual.to_numpy(), desired.numpy(), decimal=5
    )

    assert actual.is_time_series == is_time_series
    assert actual.is_voxel == is_voxel

    if dimension is not None:
        actual_dimension = actual.dimension._tensor.numpy()
        np.testing.assert_almost_equal(actual_dimension, dimension)
    else:
        assert actual.dimension is None


@pytest.mark.parametrize(
    "size, is_time_series, is_voxel",
    [
        ((10, 3, 16), False, False),
    ],
)
@pytest.mark.parametrize(
    "dimension",
    [
        [[-1], [2], [0], [0], [0], [0], [0]],
        [[1], [0], [1], [0], [0], [0], [0]],
        [[-1], [-1], [2], [0], [1], [0], [0]],
    ],
)
@pytest.mark.parametrize(
    "weight_dimension",
    [
        [[0], [0], [0], [0], [0], [0], [0]],
        [[3], [0], [1], [0], [0], [0], [0]],
    ],
)
def test_spatial_mean_w_weight(
    size: tuple[int],
    is_time_series: bool,
    is_voxel: bool,
    dimension: list[list[int]] | None,
    weight_dimension: list[list[int]] | None,
):
    torch_tensor = torch.rand(*size)
    x = phlower_tensor(
        torch_tensor,
        dimension=dimension,
        is_time_series=is_time_series,
        is_voxel=is_voxel,
    )
    weight = phlower_tensor(torch.rand(len(x), 1), dimension=dimension)
    actual = phlower.nn.functional.spatial_mean(x, weight)

    desired = torch.einsum(
        "npf,nf->pf",
        torch_tensor,
        weight.to_tensor(),
    )[None, ...] / torch.sum(weight.to_tensor())
    np.testing.assert_almost_equal(
        actual.to_numpy(), desired.numpy(), decimal=5
    )

    assert actual.is_time_series == is_time_series
    assert actual.is_voxel == is_voxel

    actual_dimension = actual.dimension._tensor.numpy()
    np.testing.assert_almost_equal(actual_dimension, np.array(dimension))


def test_inner_product():
    a = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )[None, ...]  # (1, 3, 3)-shaped tensor assuming n_feature = 3

    t = phlower_tensor(a)
    inner_product = (
        phlower.nn.functional.inner_product(t, t).reshape((3, 3)).numpy()
    )
    np.testing.assert_almost_equal(inner_product, np.eye(3))
