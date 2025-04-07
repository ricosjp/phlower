
import hypothesis.strategies as st
import numpy as np
import pytest
import torch
from hypothesis import given
from phlower import PhlowerTensor, phlower_tensor
from phlower._base import phlower_dimension_tensor
from phlower._base._functionals import to_batch
from phlower._fields import SimulationField
from phlower.collections import phlower_tensor_collection
from phlower.nn import Pooling
from phlower.settings._module_settings import PoolingSetting
from phlower.utils.enums import PoolingType


def test__can_call_parameters():
    model = Pooling(
        pool_operator_name="mean",
        nodes=[10, 10],
    )

    # To check Einsum inherit torch.nn.Module appropriately
    _ = model.parameters()


@given(
    pool_operator_name=st.sampled_from(PoolingType),
    unbatch_key=st.one_of(st.none(), st.text()),
)
def test__can_pass_parameters_via_setting(
    pool_operator_name: str,
    unbatch_key: str | None,
):
    setting = PoolingSetting(
        unbatch_key=unbatch_key, pool_operator_name=pool_operator_name
    )

    model = Pooling.from_setting(setting)

    assert model._pool_operator_name == pool_operator_name
    assert model._unbatch_key == unbatch_key


@pytest.mark.parametrize(
    "pool_operator_name",
    [
        PoolingType.max,
        PoolingType.mean,
    ],
)
@pytest.mark.parametrize(
    "input_shape, is_time_series, desired_shape",
    [
        ([10, 5], False, [1, 5]),
        ([2, 10, 3, 1], True, [2, 1, 3, 1]),
    ],
)
@pytest.mark.parametrize("uniform_value", [1.5, 0.05])
def test__pooling_for_uniform_tensor(
    pool_operator_name: PoolingType,
    input_shape: list[int],
    is_time_series: bool,
    desired_shape: list[int],
    uniform_value: float,
):
    _tensor = phlower_tensor(
        np.ones(input_shape) * uniform_value,
        is_time_series=is_time_series,
        dtype=torch.float32,
    )
    _desired_tensor = phlower_tensor(
        np.ones(desired_shape) * uniform_value,
        is_time_series=is_time_series,
        dtype=torch.float32,
    )

    model = Pooling(pool_operator_name=pool_operator_name)

    result = model.forward(phlower_tensor_collection({"input": _tensor}))

    np.testing.assert_array_almost_equal(
        result.to_tensor().numpy(), _desired_tensor.to_tensor().numpy()
    )


@pytest.mark.parametrize(
    "pool_operator_name",
    [
        PoolingType.max,
        PoolingType.mean,
    ],
)
@pytest.mark.parametrize(
    "input_shape, is_time_series, desired_shape",
    [
        ([10, 5], False, [1, 5]),
        ([2, 10, 3, 1], True, [2, 1, 3, 1]),
    ],
)
@pytest.mark.parametrize("dimension", [{}, {"T": -1, "L": 1}, None])
def test__pooling_for_not_batched_tensor(
    pool_operator_name: PoolingType,
    input_shape: list[int],
    is_time_series: bool,
    desired_shape: list[int],
    dimension: dict | None,
):
    _tensor = phlower_tensor(
        np.random.rand(*input_shape),
        is_time_series=is_time_series,
        dimension=dimension,
        dtype=torch.float32,
    )

    model = Pooling(pool_operator_name=pool_operator_name)

    result = model.forward(phlower_tensor_collection({"input": _tensor}))

    assert tuple(result.shape) == tuple(desired_shape)
    assert result.is_time_series == is_time_series

    if dimension is not None:
        assert (
            result.dimension.to_dict()
            == phlower_dimension_tensor(dimension).to_dict()
        )
    else:
        assert result.dimension is None


@pytest.mark.parametrize(
    "pool_operator_name",
    [
        PoolingType.max,
        PoolingType.mean,
    ],
)
@pytest.mark.parametrize(
    "input_shapes, is_time_series, desired_shape",
    [
        ([[10, 5], [30, 5]], False, [2, 5]),
        ([[2, 10, 3, 1], [2, 50, 3, 1]], True, [2, 2, 3, 1]),
    ],
)
@pytest.mark.parametrize("dimension", [{}, {"T": -1, "L": 1}, None])
def test__pooling_for_batched_tensor(
    pool_operator_name: PoolingType,
    input_shapes: list[list[int]],
    is_time_series: bool,
    desired_shape: list[int],
    dimension: dict | None,
):
    _tensors: list[PhlowerTensor] = [
        phlower_tensor(
            np.random.rand(*shape),
            dtype=torch.float32,
            dimension=dimension,
            is_time_series=is_time_series,
        )
        for shape in input_shapes
    ]
    _concat_dim = 1 if is_time_series else 0
    _tensor, batch_info = to_batch(_tensors, _concat_dim)
    field = SimulationField({}, {"sample": batch_info})

    model = Pooling(pool_operator_name=pool_operator_name, unbatch_key="sample")

    result = model.forward(
        phlower_tensor_collection({"input": _tensor}), field_data=field
    )
    assert tuple(result.shape) == tuple(desired_shape)
    assert result.is_time_series == is_time_series

    if dimension is not None:
        assert (
            result.dimension.to_dict()
            == phlower_dimension_tensor(dimension).to_dict()
        )
    else:
        assert result.dimension is None
