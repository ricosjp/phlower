import numpy as np
import pytest
import torch
from scipy.stats import ortho_group

from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import EnEquivariantMLP
from phlower.nn._core_modules import _functions


def test__can_call_parameters():
    model = EnEquivariantMLP(nodes=[8, 8])

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "size, is_time_series, is_voxel",
    [
        ((10, 1), False, False),
        ((10, 16), False, False),
        ((10, 3, 16), False, False),
        ((4, 10, 1), True, False),
        ((4, 10, 16), True, False),
        ((4, 10, 3, 16), True, False),
        ((10, 10, 10, 1), False, True),
        ((10, 10, 10, 16), False, True),
        ((10, 10, 10, 3, 16), False, True),
        ((4, 10, 10, 10, 1), True, True),
        ((4, 10, 10, 10, 16), True, True),
        ((4, 10, 10, 10, 3, 16), True, True),
    ],
)
@pytest.mark.parametrize("n_output_feature", [1, 16, 32])
def test__en_equivariance(
        size, is_time_series, is_voxel, n_output_feature):
    orthogonal_tensor = PhlowerTensor(
        torch.tensor(ortho_group.rvs(3).astype(np.float32)))
    create_linear_weight = size[-1] != n_output_feature
    model = EnEquivariantMLP(
        nodes=[size[-1], n_output_feature],
        create_linear_weight=create_linear_weight)

    phlower_tensor = PhlowerTensor(
        torch.rand(*size), is_time_series=is_time_series, is_voxel=is_voxel)

    phlower_tensors = phlower_tensor_collection({'tensor': phlower_tensor})
    actual = _functions.apply_orthogonal_group(
        orthogonal_tensor, model(phlower_tensors)).to_numpy()

    rotated_phlower_tensors = phlower_tensor_collection(
        {'tensor': _functions.apply_orthogonal_group(
            orthogonal_tensor, phlower_tensor)})
    desired = model(rotated_phlower_tensors).to_numpy()

    np.testing.assert_almost_equal(actual, desired, decimal=6)
