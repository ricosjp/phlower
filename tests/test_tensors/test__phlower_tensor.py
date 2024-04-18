import numpy as np
import pytest
import torch

from phlower import PhlowerTensor, phlower_dimension_tensor
from phlower.utils.exceptions import DimensionIncompatibleError


def test__add():
    a = torch.eye(5)
    b = torch.eye(5)
    c = (a + b).numpy()

    ap = PhlowerTensor(a)
    bp = PhlowerTensor(b)
    cp = ap + bp
    cp = cp.tensor().numpy()

    np.testing.assert_array_almost_equal(cp, c)


def test__add_with_unit():
    units = phlower_dimension_tensor({"length": 2, "time": -2})
    a = np.random.rand(3, 10)
    b = np.random.rand(3, 10)
    c = a + b

    ap = PhlowerTensor(torch.tensor(a), units)
    bp = PhlowerTensor(torch.tensor(b), units)
    cp = ap + bp

    np.testing.assert_array_almost_equal(cp.tensor().numpy(), c)


def test__add_with_unit_incompatible():
    units_1 = phlower_dimension_tensor({"length": 2, "time": -2})
    units_2 = phlower_dimension_tensor({"mass": 1, "time": -2})

    a = torch.from_numpy(np.random.rand(3, 10))
    b = torch.from_numpy(np.random.rand(3, 10))
    with pytest.raises(DimensionIncompatibleError):
        ap = PhlowerTensor(a, units_1)
        bp = PhlowerTensor(b, units_2)
        _ = ap + bp


def test__mul_with_unit():
    dims_1 = phlower_dimension_tensor({"length": 2, "time": -2})
    dims_2 = phlower_dimension_tensor({"mass": 1, "time": -2})
    dims_3 = phlower_dimension_tensor({"length": 2, "mass": 1, "time": -4})

    a = torch.tensor(np.random.rand(3, 10))
    b = torch.tensor(np.random.rand(3, 10))
    c = a * b

    ap = PhlowerTensor(a, dims_1)
    bp = PhlowerTensor(b, dims_2)
    cp = ap * bp

    np.testing.assert_array_almost_equal(cp.tensor().numpy(), c.numpy())

    assert cp._dimension_tensor == dims_3


def test__tanh():
    a = torch.tensor(np.random.rand(3, 10))
    c = np.tanh(a)

    ap = PhlowerTensor(a)
    cp = torch.tanh(ap)

    np.testing.assert_array_almost_equal(cp.tensor(), c)


# @pytest.mark.parametrize("args, attr, expect", [
#     (
#         [
#             phlower_dimension_tensor({"length": 2, "time": -2}),
#             phlower_dimension_tensor({"length": 2, "time": -2}),
#             phlower_dimension_tensor({"length": 2, "time": -2}),
#         ],
#         "_"
#     )
# ])
# def test__has_dimension():
#     ...


@pytest.mark.parametrize(
    "key",
    [
        3,
        [1, 3, 4],
        [True, False, False, True, True],
    ],
)
def test__getitem(key):
    torch_tensor = torch.rand(5)
    phlower_tensor = PhlowerTensor(torch_tensor)
    np.testing.assert_array_almost_equal(
        phlower_tensor[key].tensor(), torch_tensor[key])
