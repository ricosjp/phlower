import numpy as np
import pytest
import torch

from phlower.tensors import PhysicsTensor, physics_unit_tensor
from phlower.utils.exceptions import UnitIncompatibleError


def test__add():
    a = torch.eye(5)
    b = torch.eye(5)
    c = (a + b).numpy()

    ap = PhysicsTensor(a)
    bp = PhysicsTensor(b)
    cp = ap + bp
    cp = cp.tensor().numpy()

    np.testing.assert_array_almost_equal(cp, c)


def test__add_with_unit():
    units = physics_unit_tensor({"m": 2, "s": -2})
    a = np.random.rand(3, 10)
    b = np.random.rand(3, 10)
    c = a + b

    ap = PhysicsTensor(torch.tensor(a), units)
    bp = PhysicsTensor(torch.tensor(b), units)
    cp = ap + bp

    np.testing.assert_array_almost_equal(cp.tensor().numpy(), c)


def test__add_with_unit_incompatible():
    units_1 = physics_unit_tensor({"m": 2, "s": -2})
    units_2 = physics_unit_tensor({"kg": 1, "s": -2})

    a = np.random.rand(3, 10)
    b = np.random.rand(3, 10)
    with pytest.raises(UnitIncompatibleError):
        ap = PhysicsTensor(a, units_1)
        bp = PhysicsTensor(b, units_2)
        _ = ap + bp


def test__mul_with_unit():
    units_1 = physics_unit_tensor({"m": 2, "s": -2})
    units_2 = physics_unit_tensor({"kg": 1, "s": -2})
    units_3 = physics_unit_tensor({"m": 2, "kg": 1, "s": -4})

    a = np.random.rand(3, 10)
    b = np.random.rand(3, 10)
    c = a * b

    ap = PhysicsTensor(a, units_1)
    bp = PhysicsTensor(b, units_2)
    cp = ap * bp

    np.testing.assert_array_almost_equal(
        cp.tensor().numpy(),
        c
    )

    assert cp._unit_tensor == units_3
