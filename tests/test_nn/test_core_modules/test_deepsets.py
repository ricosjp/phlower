from collections.abc import Callable

import hypothesis.strategies as st
import numpy as np
import pytest
import torch
from hypothesis import given
from phlower import PhlowerTensor, phlower_tensor
from phlower._base import phlower_dimension_tensor
from phlower._base._functionals import is_same_dimensions, to_batch, unbatch
from phlower._fields import SimulationField
from phlower.collections import phlower_tensor_collection
from phlower.nn import DeepSets, MLPConfiguration
from phlower.settings._module_settings import DeepSetsSetting
from phlower.utils.enums import ActivationType


def test__can_call_parameters():
    config = MLPConfiguration(nodes=[10, 10], activations=[], dropouts=[])
    model = DeepSets(
        lambda_config=config,
        gamma_config=config,
        last_activation_name="identity",
        pool_operator_name="max",
    )

    # To check Einsum inherit torch.nn.Module appropriately
    _ = model.parameters()


@st.composite
def create_mlp_configs(
    draw: Callable,
) -> tuple[list[int], list[str], list[float], bool]:
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
    activations = [activation.name for _ in range(n_items)]
    dropouts = draw(
        st.lists(
            st.floats(width=32, min_value=0, max_value=1.0, exclude_max=True),
            min_size=n_items,
            max_size=n_items,
        )
    )
    bias = draw(st.booleans())

    return nodes, activations, dropouts, bias


@pytest.mark.parametrize("pool_operator_name", ["max", "mean"])
@given(
    last_activation=st.sampled_from(ActivationType),
    mlp_configs=create_mlp_configs(),
)
def test__can_pass_parameters_via_setting(
    pool_operator_name: str,
    last_activation: ActivationType,
    mlp_configs: tuple[list[int], list[str], list[float], bool],
):
    nodes, activations, dropouts, bias = mlp_configs

    setting = DeepSetsSetting(
        last_activation=last_activation.name,
        pool_operator=pool_operator_name,
        nodes=nodes,
        activations=activations,
        dropouts=dropouts,
        bias=bias,
    )
    model = DeepSets.from_setting(setting)

    assert model._last_activation_name == last_activation.name
    assert model._pool_operator_name == pool_operator_name

    assert model._lambda._nodes == nodes
    assert model._lambda._activations == activations
    assert model._lambda._dropouts == dropouts
    assert model._lambda.has_bias() is bias

    assert model._gamma._nodes == nodes
    assert model._gamma._activations == activations
    assert model._gamma._dropouts == dropouts
    assert model._gamma.has_bias() is bias


@pytest.mark.parametrize(
    "nodes, activations, bias, last_activation_name, pool_operator_name",
    [
        ([-1, 10, 5], ["relu", "relu"], True, "relu", "max"),
        ([-1, 10, 20], ["relu", "identity"], True, "tanh", "mean"),
    ],
)
@pytest.mark.parametrize(
    "input_shape, is_time_series",
    [
        ((10, 3, 1), False),
        ((20, 8), False),
        ((5, 20, 2, 1), False),
        ((2, 10, 3, 1), True),
        ((1, 20, 6), True),
        ((5, 100, 5), True),
    ],
)
@pytest.mark.parametrize(
    "dimension", [{"Theta": 1}, {"M": 1, "T": -1}, None, {}]
)
def test__permutation_equivariance_for_vertexwise_tensor(
    nodes: list[int],
    activations: list[str],
    bias: bool,
    last_activation_name: str,
    pool_operator_name: str,
    input_shape: tuple[int],
    is_time_series: bool,
    dimension: dict | None,
):
    nodes[0] = input_shape[-1]
    _tensor = phlower_tensor(
        np.random.rand(*input_shape),
        is_time_series=is_time_series,
        dimension=dimension,
    )

    config = MLPConfiguration(
        nodes=nodes, activations=activations, dropouts=[], bias=bias
    )
    model = DeepSets(
        lambda_config=config,
        gamma_config=config,
        last_activation_name=last_activation_name,
        pool_operator_name=pool_operator_name,
    )

    result1 = model.forward(phlower_tensor_collection({"input": _tensor}))
    assert len(result1.shape) == len(input_shape)
    assert result1.shape[-1] == nodes[-1]
    if dimension is not None:
        assert (
            result1.dimension.to_dict()
            == phlower_dimension_tensor(dimension).to_dict()
        )
    else:
        assert result1.dimension is None

    n_nodes = _tensor.shape_pattern.get_n_vertices()
    permute_idxes = np.random.permutation(n_nodes)

    # Permutation along node axis
    def _random_permutation(
        value: PhlowerTensor, indexes: np.ndarray
    ) -> torch.Tensor:
        if is_time_series:
            return value[:, indexes, ...]
        return value[indexes, ...]

    _permuted_tensor = phlower_tensor(
        _random_permutation(_tensor, permute_idxes),
        dimension=dimension,
        is_time_series=is_time_series,
    )
    result2 = model.forward(
        phlower_tensor_collection({"input": _permuted_tensor})
    )

    np.testing.assert_array_almost_equal(
        _random_permutation(result1, permute_idxes).detach().numpy(),
        result2.to_numpy(),
    )

    assert is_same_dimensions([result1, result2])


@pytest.mark.parametrize(
    "nodes, activations, bias, last_activation_name, pool_operator_name",
    [
        ([-1, 10, 5], ["tanh", "tanh"], True, "tanh", "max"),
        ([-1, 10, 20], ["relu", "tanh"], True, "relu", "mean"),
    ],
)
@pytest.mark.parametrize(
    "input_shape, is_time_series",
    [
        ((10, 3, 1), False),
        ((20, 8), False),
        ((5, 20, 2, 1), False),
        ((2, 10, 3, 1), True),
        ((1, 20, 6), True),
        ((5, 100, 5), True),
    ],
)
@pytest.mark.parametrize("dimension", [{}, None])
def test__permutation_equivariance_with_non_dimension(
    nodes: list[int],
    activations: list[str],
    bias: bool,
    last_activation_name: str,
    pool_operator_name: str,
    input_shape: tuple[int],
    is_time_series: bool,
    dimension: dict | None,
):
    nodes[0] = input_shape[-1]
    _tensor = phlower_tensor(
        np.random.rand(*input_shape),
        is_time_series=is_time_series,
        dimension=dimension,
    )

    config = MLPConfiguration(
        nodes=nodes, activations=activations, dropouts=[], bias=bias
    )
    model = DeepSets(
        lambda_config=config,
        gamma_config=config,
        last_activation_name=last_activation_name,
        pool_operator_name=pool_operator_name,
    )

    result1 = model.forward(phlower_tensor_collection({"input": _tensor}))
    assert len(result1.shape) == len(input_shape)
    assert result1.shape[-1] == nodes[-1]
    if dimension is not None:
        assert (
            result1.dimension.to_dict()
            == phlower_dimension_tensor(dimension).to_dict()
        )
    else:
        assert result1.dimension is None

    n_nodes = _tensor.shape_pattern.get_n_vertices()
    permute_idxes = np.random.permutation(n_nodes)

    # Permutation along node axis
    def _random_permutation(
        value: PhlowerTensor, indexes: np.ndarray
    ) -> torch.Tensor:
        if is_time_series:
            return value[:, indexes, ...]
        return value[indexes, ...]

    _permuted_tensor = phlower_tensor(
        _random_permutation(_tensor, permute_idxes),
        dimension=dimension,
        is_time_series=is_time_series,
    )
    result2 = model.forward(
        phlower_tensor_collection({"input": _permuted_tensor})
    )

    np.testing.assert_array_almost_equal(
        _random_permutation(result1, permute_idxes).detach().numpy(),
        result2.to_numpy(),
    )

    assert is_same_dimensions([result1, result2])


@pytest.mark.parametrize(
    "nodes, activations, bias, last_activation_name, pool_operator_name",
    [
        ([-1, 10, 5], ["tanh", "tanh"], True, "relu", "max"),
        ([-1, 10, 20], ["relu", "identity"], True, "tanh", "mean"),
    ],
)
@pytest.mark.parametrize(
    "input_shapes, desired_shape",
    [([(10, 3, 1), (20, 3, 1)], (30, 3, 1))],
)
def test__permutation_equivariance_for_batched_tensor(
    nodes: list[int],
    activations: list[str],
    bias: bool,
    last_activation_name: str,
    pool_operator_name: str,
    input_shapes: list[tuple[int]],
    desired_shape: tuple[int],
):
    nodes[0] = input_shapes[0][-1]

    _tensors: list[PhlowerTensor] = [
        phlower_tensor(np.random.rand(*shape)) for shape in input_shapes
    ]
    _tensor, batch_info = to_batch(_tensors)
    field = SimulationField({}, {"sample": batch_info})

    config = MLPConfiguration(
        nodes=nodes, activations=activations, dropouts=[], bias=bias
    )
    model = DeepSets(
        lambda_config=config,
        gamma_config=config,
        last_activation_name=last_activation_name,
        pool_operator_name=pool_operator_name,
    )

    result1 = model.forward(
        phlower_tensor_collection({"input": _tensor}), field_data=field
    )
    assert len(result1.shape) == len(desired_shape)
    assert result1.shape[-1] == nodes[-1]

    # Calculate for randomly permutated tensor
    _permuted_indexes = [
        np.random.permutation(t.shape_pattern.get_n_vertices())
        for t in _tensors
    ]
    _permuted_tensors = [
        phlower_tensor(t[_permuted_indexes[i], ...])
        for i, t in enumerate(_tensors)
    ]
    _permuted, _ = to_batch(_permuted_tensors)
    result2 = model.forward(
        phlower_tensor_collection({"input": _permuted}), field_data=field
    )

    _unbatched_result2 = unbatch(result2, batch_info)
    for i, result in enumerate(unbatch(result1, batch_info=batch_info)):
        np.testing.assert_array_almost_equal(
            result[_permuted_indexes[i], ...].detach().numpy(),
            _unbatched_result2[i].to_tensor().detach().numpy(),
        )


@pytest.mark.parametrize(
    "nodes, activations, bias, last_activation_name, pool_operator_name",
    [
        ([-1, 10, 5], ["tanh", "tanh"], True, "relu", "max"),
        ([-1, 10, 20], ["relu", "identity"], True, "tanh", "mean"),
    ],
)
@pytest.mark.parametrize(
    "input_shapes, desired_shape",
    [([(10, 3, 1), (20, 3, 1)], (30, 3, 1))],
)
def test__permutation_NOT_equivariance_over_different_batch_tensor(
    nodes: list[int],
    activations: list[str],
    bias: bool,
    last_activation_name: str,
    pool_operator_name: str,
    input_shapes: list[tuple[int]],
    desired_shape: tuple[int],
):
    nodes[0] = input_shapes[0][-1]

    _tensors: list[PhlowerTensor] = [
        phlower_tensor(np.random.rand(*shape)) for shape in input_shapes
    ]
    _tensor, batch_info = to_batch(_tensors)
    field = SimulationField({}, {"sample": batch_info})

    config = MLPConfiguration(
        nodes=nodes, activations=activations, dropouts=[], bias=bias
    )
    model = DeepSets(
        lambda_config=config,
        gamma_config=config,
        last_activation_name=last_activation_name,
        pool_operator_name=pool_operator_name,
    )

    result1 = model.forward(
        phlower_tensor_collection({"input": _tensor}), field_data=field
    )
    assert len(result1.shape) == len(desired_shape)
    assert result1.shape[-1] == nodes[-1]

    # Calculate for randomly permutated tensor
    # indexes are completely random across batch region intentionally.
    _permuted_indexes = np.random.permutation(
        _tensor.shape_pattern.get_n_vertices()
    )
    _permuted = _tensor[_permuted_indexes, ...]
    result2 = model.forward(
        phlower_tensor_collection({"input": _permuted}), field_data=field
    )

    with pytest.raises(AssertionError) as ex:
        np.testing.assert_array_almost_equal(
            result1[_permuted_indexes, ...].detach().numpy(),
            result2.to_tensor().detach().numpy(),
        )

    assert "Arrays are not almost equal to 6 decimals" in str(ex.value)
