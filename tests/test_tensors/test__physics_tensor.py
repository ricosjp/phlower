import pytest

import torch
import numpy as np

from phlower.tensors import PhysicsTensor


def test__add():
    a = torch.eye(5)
    b = torch.eye(5)
    ap = PhysicsTensor(a)
    bp = PhysicsTensor(b)

    cp = ap + bp
    cp = cp._tensor.numpy()
    c = (a + b).numpy()

    print(cp)
    np.testing.assert_array_almost_equal(
        cp, c
    )
