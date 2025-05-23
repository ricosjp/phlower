from collections.abc import Callable

import hypothesis.strategies as st
import numpy as np
import pytest
import torch
from hypothesis import given
from phlower import phlower_tensor
from phlower._base import PhysicalDimensions
from phlower.collections import phlower_tensor_collection
from phlower.nn import TCN
from phlower.settings._module_settings import TCNSetting
from phlower.utils.enums import ActivationType


def test__can_call_parameters():
    model = TCN(
        nodes=[10, 10],
        kernel_sizes=[3],
        activations=[],
        dropouts=[],
        dilations=[],
        bias=True,
    )

    # To check Einsum inherit torch.nn.Module appropriately
    _ = model.parameters()


@st.composite
def create_tcn_setting_items(
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
    tcn_configs=create_tcn_setting_items(),
)
def test__can_pass_parameters_via_setting(
    tcn_configs: tuple[list[int], list[str], list[float], bool],
):
    nodes, kernel_sizes, dilations, activations, dropouts, bias = tcn_configs

    setting = TCNSetting(
        nodes=nodes,
        activations=activations,
        dropouts=dropouts,
        bias=bias,
        kernel_sizes=kernel_sizes,
        dilations=dilations,
    )
    model = TCN.from_setting(setting)

    assert model._convs._nodes == nodes
    assert model._convs._activations == activations
    assert model._convs._dropouts == dropouts
    assert model._convs._bias is bias
    assert model._convs._kernel_sizes == kernel_sizes
    assert model._convs._dilations == dilations


def test__default_values():
    model = TCN(
        nodes=[10, 10],
        kernel_sizes=[3],
        activations=[],
        dilations=[],
    )
    assert model._convs._activations == ["identity"]
    assert model._convs._dropouts == [0.0]
    assert model._convs._dilations == [1]
    assert model._convs._bias


@pytest.mark.parametrize(
    "nodes, activations, kernel_sizes, dilations",
    [
        ([-1, 5, 10], ["identity", "identity"], [3, 3], [1, 2]),
        ([-1, 10, 20], ["relu", "identity"], [4, 5], [1, 4]),
    ],
)
@pytest.mark.parametrize(
    "input_shape, is_voxel",
    [
        ((4, 10, 1), False),
        ((5, 10, 16), False),
        ((6, 10, 3, 16), False),
        ((1, 10, 10, 10, 1), True),
        ((3, 10, 10, 10, 16), True),
        ((2, 10, 10, 10, 3, 16), True),
    ],
)
@pytest.mark.parametrize("dimension", [{"Theta": 1}, {"M": 1, "T": -1}, None])
def test__output_tensor_shape(
    nodes: list[int],
    activations: list[str],
    kernel_sizes: list[int],
    dilations: list[int],
    input_shape: tuple[int],
    is_voxel: bool,
    dimension: dict | None,
):
    nodes[0] = input_shape[-1]
    _tensor = phlower_tensor(
        np.random.rand(*input_shape),
        is_time_series=True,
        dimension=dimension,
        is_voxel=is_voxel,
        dtype=torch.float32,
    )

    model = TCN(
        nodes=nodes,
        kernel_sizes=kernel_sizes,
        dilations=dilations,
        activations=activations,
    )

    actual = model.forward(phlower_tensor_collection({"input": _tensor}))

    assert actual.is_time_series

    desired_shape = [*input_shape[:-1], nodes[-1]]
    assert tuple(actual.shape) == tuple(desired_shape)

    if _tensor.dimension:
        assert actual.dimension.to_physics_dimension() == PhysicalDimensions(
            dimension
        )
    else:
        assert actual.dimension is None


@pytest.mark.parametrize(
    "shape, is_voxel", [((10, 1), False), ((5, 5, 5, 8), True)]
)
def test__raise_error_when_input_is_not_time_series(
    shape: tuple[int], is_voxel: bool
):
    model = TCN(
        nodes=[10, 10],
        kernel_sizes=[2],
        activations=[],
        dilations=[],
    )

    inputs = phlower_tensor_collection(
        {
            "inputs": phlower_tensor(
                np.random.rand(*shape), is_voxel=is_voxel, dtype=torch.float32
            )
        }
    )

    with pytest.raises(ValueError) as ex:
        model.forward(inputs)

    assert "Input tensor to TCN is not time-series tensor" in str(ex.value)
