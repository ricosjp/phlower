import numpy as np
import pytest
import torch

from phlower.tensors import PhlowerTensor, physical_dimension_tensor
from phlower.utils.exceptions import UnitIncompatibleError


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
    units = physical_dimension_tensor({"length": 2, "time": -2})
    a = np.random.rand(3, 10)
    b = np.random.rand(3, 10)
    c = a + b

    ap = PhlowerTensor(torch.tensor(a), units)
    bp = PhlowerTensor(torch.tensor(b), units)
    cp = ap + bp

    np.testing.assert_array_almost_equal(cp.tensor().numpy(), c)


def test__add_with_unit_incompatible():
    units_1 = physical_dimension_tensor({"length": 2, "time": -2})
    units_2 = physical_dimension_tensor({"mass": 1, "time": -2})

    a = np.random.rand(3, 10)
    b = np.random.rand(3, 10)
    with pytest.raises(UnitIncompatibleError):
        ap = PhlowerTensor(a, units_1)
        bp = PhlowerTensor(b, units_2)
        _ = ap + bp


def test__mul_with_unit():
    units_1 = physical_dimension_tensor({"length": 2, "time": -2})
    units_2 = physical_dimension_tensor({"mass": 1, "time": -2})
    units_3 = physical_dimension_tensor({"length": 2, "mass": 1, "time": -4})

    a = np.random.rand(3, 10)
    b = np.random.rand(3, 10)
    c = a * b

    ap = PhlowerTensor(a, units_1)
    bp = PhlowerTensor(b, units_2)
    cp = ap * bp

    np.testing.assert_array_almost_equal(cp.tensor().numpy(), c)

    assert cp._unit_tensor == units_3


def test__tanh():
    a = np.random.rand(3, 10)
    c = np.tanh(a)

    ap = PhlowerTensor(a)
    cp = torch.tanh(ap)

    np.testing.assert_array_almost_equal(cp.tensor(), c)
