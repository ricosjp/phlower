import numpy as np
import pytest
import torch
from scipy.stats import ortho_group

from phlower import PhlowerTensor, phlower_tensor
from phlower._base._dimension import PhysicalDimensions
from phlower.collections import phlower_tensor_collection
from phlower.nn import SimilarityEquivariantMLP
from phlower.nn._core_modules import _functions
from phlower.utils.exceptions import (
    PhlowerDimensionRequiredError,
    PhlowerInvalidArgumentsError,
)


def test__can_call_parameters():
    model = SimilarityEquivariantMLP(nodes=[8, 8])

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
@pytest.mark.parametrize("activation", ["identity", "tanh", "leaky_relu0p5"])
@pytest.mark.parametrize("n_output_feature", [1, 16, 32])
@pytest.mark.parametrize(
    "dimension",
    [
        # Dimensionless
        {"T": 0, "L": 0, "M": 0, "I": 0, "Theta": 0, "N": 0, "J": 0},
        # Velocity
        {"T": -1, "L": 1, "M": 0, "I": 0, "Theta": 0, "N": 0, "J": 0},
        # Mass density
        {"T": 0, "L": - 3, "M": 1, "I": 0, "Theta": 0, "N": 0, "J": 0},
        # Momentum density
        {"T": -1, "L": 1 - 3, "M": 1, "I": 0, "Theta": 0, "N": 0, "J": 0},
    ],
)
@pytest.mark.parametrize(
    "norm_function_name", ["identity", "sqrt"])
@pytest.mark.parametrize(
    "disable_en_equivariance", [False, True])
@pytest.mark.parametrize(
    "centering", [False, True])
@pytest.mark.parametrize(
    "invariant", [False, True])
def test__similarity_equivariance(
        size, is_time_series, is_voxel, activation, n_output_feature,
        dimension, norm_function_name, disable_en_equivariance,
        centering, invariant,
):
    if disable_en_equivariance:
        orthogonal_tensor = PhlowerTensor(torch.eye(3))
    else:
        orthogonal_tensor = PhlowerTensor(
            torch.tensor(ortho_group.rvs(3).astype(np.float32)))
    dict_scaling_factor = {
        k:
        np.random.rand() * 10 for k, v in dimension.items()}
    scaling_factor = np.prod(
        [dict_scaling_factor[k]**v for k, v in dimension.items()])

    create_linear_weight = size[-1] != n_output_feature
    model = SimilarityEquivariantMLP(
        nodes=[size[-1], n_output_feature, n_output_feature],
        activations=[activation, activation],
        create_linear_weight=create_linear_weight,
        norm_function_name=norm_function_name,
        disable_en_equivariance=disable_en_equivariance,
        centering=centering, invariant=invariant,
    )

    t = phlower_tensor(
        torch.randn(*size), dimension=dimension,
        is_time_series=is_time_series, is_voxel=is_voxel)
    dict_tensor = {"tensor": t}
    scales = {
        k: phlower_tensor(
            torch.rand(1) * 10, dimension=PhysicalDimensions({k: 1}))
        for k, v in dimension.items()
        if v != 0
    }
    if len(scales) == 0:
        scales = {"L": phlower_tensor(
            torch.rand(1) * 10, dimension=PhysicalDimensions({"L": 1}))}
    dict_tensor.update(scales)

    ts = phlower_tensor_collection(dict_tensor)
    if invariant:
        actual_tensor = _functions.apply_orthogonal_group(
            orthogonal_tensor, model(ts)) * 1.
    else:
        actual_tensor = _functions.apply_orthogonal_group(
            orthogonal_tensor, model(ts)) * scaling_factor
    actual = actual_tensor.to_numpy()

    dict_transformed = {'tensor': _functions.apply_orthogonal_group(
            orthogonal_tensor, t) * scaling_factor}
    dict_scaled_scales = {
        k: v * dict_scaling_factor[k] for k, v in scales.items()}
    dict_transformed.update(dict_scaled_scales)

    transformed_phlower_tensors = phlower_tensor_collection(dict_transformed)
    desired = model(transformed_phlower_tensors)

    norm = torch.mean(_functions.contraction(desired)**.5).to_numpy()
    # Test equivariance
    np.testing.assert_almost_equal(
        actual / norm, desired.to_numpy() / norm, decimal=2)

    if invariant:
        # Test dimensionless in case of invariant
        for v in actual_tensor.dimension.to_dict().values():
            np.testing.assert_almost_equal(v, 0.)
    else:
        # Test dimension is kept
        for k, v in actual_tensor.dimension.to_dict().items():
            np.testing.assert_almost_equal(v, dimension[k])


def test__similarity_equivariance_no_dimension():
    model = SimilarityEquivariantMLP(nodes=[8, 8])

    t = phlower_tensor(torch.rand(10, 3, 8), dimension=None)
    t_scale = phlower_tensor(
        torch.tensor(0.1), dimension=PhysicalDimensions({"T": 1}))
    ts = phlower_tensor_collection({"tensor": t, "T": t_scale})
    with pytest.raises(PhlowerDimensionRequiredError):
        model(ts)

def test__similarity_equivariance_no_scale_input():
    model = SimilarityEquivariantMLP(nodes=[8, 8])
    dimension = {"T": -1, "L": 1, "M": 0, "I": 0, "Theta": 0, "N": 0, "J": 0}

    t = phlower_tensor(torch.rand(10, 3, 8), dimension=dimension)
    ts = phlower_tensor_collection({"tensor": t})
    with pytest.raises(PhlowerInvalidArgumentsError):
        model(ts)
