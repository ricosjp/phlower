import numpy as np
import pytest
import torch
from scipy import sparse as sp

from phlower import PhlowerTensor
from phlower.nn._core_modules import _functions


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
def test__spmm(size, is_time_series, repeat):
    phlower_tensor = PhlowerTensor(
        torch.rand(*size), is_time_series=is_time_series
    )
    n = phlower_tensor.n_vertices()
    sparse = PhlowerTensor(torch.rand(n, n).to_sparse())

    actual_spmm = (
        _functions.spmm(sparse, phlower_tensor, repeat=repeat)
        .to_tensor()
        .numpy()
    )
    sp_sparse = sp.coo_array(sparse.to_tensor().to_dense().numpy())
    np_dense = phlower_tensor.to_tensor().numpy()

    def assert_correct(actual, array):
        dim_feat = len(array.shape) - 1
        if dim_feat == 1:
            desired = array
            for _ in range(repeat):
                desired = sp_sparse @ desired
            norm = np.mean(np.linalg.norm(desired, axis=-1))
            np.testing.assert_almost_equal(
                actual / norm, desired / norm, decimal=5)
            return

        for i in range(array.shape[1]):
            assert_correct(actual[:, i], array[:, i])
        return

    if is_time_series:
        for t in range(size[0]):
            assert_correct(actual_spmm[t], np_dense[t])
    else:
        assert_correct(actual_spmm, np_dense)
