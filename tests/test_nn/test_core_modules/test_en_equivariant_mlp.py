import numpy as np
import pytest
import torch
from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import EnEquivariantMLP
from phlower.nn._functionals import _functions
from scipy.stats import ortho_group


def assert_en_equivariance(
    size: tuple[int],
    is_time_series: bool,
    is_voxel: bool,
    norm_function_name: str,
    cross_interaction: bool,
    normalize: bool,
    n_output_feature: int,
):
    orthogonal_tensor = PhlowerTensor(
        torch.tensor(ortho_group.rvs(3).astype(np.float32))
    )
    create_linear_weight = size[-1] != n_output_feature
    model = EnEquivariantMLP(
        nodes=[size[-1], n_output_feature],
        create_linear_weight=create_linear_weight,
        norm_function_name=norm_function_name,
        cross_interaction=cross_interaction,
        normalize=normalize,
    )

    phlower_tensor = PhlowerTensor(
        torch.randn(*size), is_time_series=is_time_series, is_voxel=is_voxel
    )

    phlower_tensors = phlower_tensor_collection({"tensor": phlower_tensor})
    actual = _functions.apply_orthogonal_group(
        orthogonal_tensor, model(phlower_tensors)
    ).to_numpy()

    assert not np.any(np.isnan(actual))

    rotated_phlower_tensors = phlower_tensor_collection(
        {
            "tensor": _functions.apply_orthogonal_group(
                orthogonal_tensor, phlower_tensor
            )
        }
    )
    desired = model(rotated_phlower_tensors).to_numpy()

    np.testing.assert_almost_equal(actual, desired, decimal=4)


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
@pytest.mark.parametrize("norm_function_name", ["identity", "sqrt"])
@pytest.mark.parametrize("cross_interaction", [False])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("n_output_feature", [1, 16, 32])
def test__en_equivariance_wo_cross_interaction(
    size: tuple[int],
    is_time_series: bool,
    is_voxel: bool,
    norm_function_name: str,
    cross_interaction: bool,
    normalize: bool,
    n_output_feature: int,
):
    assert_en_equivariance(
        size=size,
        is_time_series=is_time_series,
        is_voxel=is_voxel,
        norm_function_name=norm_function_name,
        cross_interaction=cross_interaction,
        normalize=normalize,
        n_output_feature=n_output_feature,
    )


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
@pytest.mark.parametrize("cross_interaction", [True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("n_output_feature", [1, 16, 32])
def test__en_equivariance_w_cross_interaction(
    size: tuple[int],
    is_time_series: bool,
    is_voxel: bool,
    cross_interaction: bool,
    normalize: bool,
    n_output_feature: int,
):
    assert_en_equivariance(
        size=size,
        is_time_series=is_time_series,
        is_voxel=is_voxel,
        norm_function_name="identity",
        cross_interaction=cross_interaction,
        normalize=normalize,
        n_output_feature=n_output_feature,
    )


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
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("n_output_feature", [1, 16, 32])
def test__en_equivariance_raise_when_sqrt_cross_intersection(
    size: tuple[int],
    is_time_series: bool,
    is_voxel: bool,
    normalize: bool,
    n_output_feature: int,
):
    norm_function_name = "sqrt"
    cross_interaction = True

    create_linear_weight = size[-1] != n_output_feature
    with pytest.raises(
        ValueError,
        match="Cannot set norm_function_name to sqrt when cross_interaction "
        "is True.",
    ):
        EnEquivariantMLP(
            nodes=[size[-1], n_output_feature],
            create_linear_weight=create_linear_weight,
            norm_function_name=norm_function_name,
            cross_interaction=cross_interaction,
            normalize=normalize,
        )
