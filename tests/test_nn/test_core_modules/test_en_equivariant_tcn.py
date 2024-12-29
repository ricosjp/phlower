from collections.abc import Callable

import hypothesis.strategies as st
import numpy as np
import pytest
import torch
from hypothesis import given
from phlower import phlower_tensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import EnEquivariantTCN
from phlower.nn._core_modules import _functions
from phlower.settings._module_settings import EnEquivariantTCNSetting
from phlower.utils.enums import ActivationType
from scipy.stats import ortho_group


def test__can_call_parameters():
    model = EnEquivariantTCN(
        nodes=[10, 10], kernel_sizes=[3], dilations=[], activations=[]
    )

    # To check Einsum inherit torch.nn.Module appropriately
    _ = model.parameters()


@st.composite
def create_en_tcn_setting_items(
    draw: Callable,
) -> tuple[list[int], list[int], list[int], list[str], list[float], bool]:
    n_nodes = draw(st.integers(min_value=2, max_value=5))
    nodes = draw(
        st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=n_nodes,
            max_size=n_nodes,
        )
    )

    activation = draw(st.sampled_from(ActivationType))
    n_items = n_nodes - 1
    kernel_sizes = draw(
        st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=n_items,
            max_size=n_items,
        )
    )
    dilations = draw(
        st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=n_items,
            max_size=n_items,
        )
    )
    activations = [activation.name for _ in range(n_items)]
    dropouts = draw(
        st.lists(
            st.floats(width=32, min_value=0, max_value=1.0, exclude_max=True),
            min_size=n_items,
            max_size=n_items,
        )
    )
    bias = draw(st.booleans())

    return nodes, kernel_sizes, dilations, activations, dropouts, bias


@given(
    tcn_configs=create_en_tcn_setting_items(),
)
def test__can_pass_parameters_via_setting(
    tcn_configs: tuple[list[int], list[str], list[float], bool],
):
    nodes, kernel_sizes, dilations, activations, dropouts, bias = tcn_configs

    setting = EnEquivariantTCNSetting(
        nodes=nodes,
        activations=activations,
        dropouts=dropouts,
        bias=bias,
        kernel_sizes=kernel_sizes,
        dilations=dilations,
        create_linear_weight=True,
    )
    model = EnEquivariantTCN.from_setting(setting)

    assert model._tcn._convs._nodes == nodes
    assert model._tcn._convs._activations == activations
    assert model._tcn._convs._dropouts == dropouts
    assert model._tcn._convs._bias is bias
    assert model._tcn._convs._kernel_sizes == kernel_sizes
    assert model._tcn._convs._dilations == dilations


@pytest.mark.parametrize(
    "nodes, kernel_sizes, dilations",
    [
        ([-1, 10, 10], [3, 3], []),
        ([-1, 10, 20], [5, 5], [1, 2]),
        ([-1, 5, 5, 5], [3, 3, 3], [1, 2, 4]),
    ],
)
@pytest.mark.parametrize(
    "shape, is_voxel",
    [
        ((4, 10, 1), False),
        ((4, 10, 16), False),
        ((4, 10, 3, 16), False),
        ((4, 10, 10, 10, 1), True),
        ((4, 10, 10, 10, 16), True),
        ((4, 10, 10, 10, 3, 16), True),
    ],
)
@pytest.mark.parametrize("dimension", [{"Theta": 1}, {"M": 1, "T": -1}, None])
def test__en_equivariance(
    nodes: list[int],
    kernel_sizes: list[int],
    dilations: list[int],
    shape: tuple[int],
    is_voxel: bool,
    dimension: dict | None,
):
    ortho_dimension = {} if dimension else None
    orthogonal_tensor = phlower_tensor(
        ortho_group.rvs(3).astype(np.float32), dimension=ortho_dimension
    )
    nodes[0] = shape[-1]
    create_linear_weight = nodes[0] != nodes[-1]
    model = EnEquivariantTCN(
        nodes=nodes,
        kernel_sizes=kernel_sizes,
        activations=[],
        dilations=dilations,
        create_linear_weight=create_linear_weight,
    )

    input_tensor = phlower_tensor(
        torch.rand(*shape),
        is_time_series=True,
        is_voxel=is_voxel,
        dimension=dimension,
    )

    phlower_tensors = phlower_tensor_collection({"tensor": input_tensor})
    actual = _functions.apply_orthogonal_group(
        orthogonal_tensor, model.forward(phlower_tensors)
    )

    rotated_phlower_tensors = phlower_tensor_collection(
        {
            "tensor": _functions.apply_orthogonal_group(
                orthogonal_tensor, input_tensor
            )
        }
    )
    desired = model.forward(rotated_phlower_tensors)

    if dimension:
        assert actual.dimension == desired.dimension

    np.testing.assert_array_almost_equal(
        actual.to_numpy(), desired.to_numpy(), decimal=6
    )


@pytest.mark.parametrize(
    "shape, is_voxel", [((10, 1), False), ((5, 5, 5, 8), True)]
)
def test__raise_error_when_input_is_not_time_series(
    shape: tuple[int], is_voxel: bool
):
    model = EnEquivariantTCN(
        nodes=[10, 10],
        kernel_sizes=[2],
        activations=[],
        dilations=[],
    )

    inputs = phlower_tensor_collection(
        {"inputs": phlower_tensor(torch.rand(*shape), is_voxel=is_voxel)}
    )

    with pytest.raises(ValueError) as ex:
        model.forward(inputs)

    assert "Only time series tensor is allowed" in str(ex.value)
