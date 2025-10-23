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
from phlower.nn import TransolverAttention
from phlower.settings._module_settings import TransolverAttentionSetting


def test__can_call_parameters():
    model = TransolverAttention(nodes=[16, 16, 16], heads=8)

    # To check TransolverAttention inherit torch.nn.Module appropriately
    _ = model.parameters()


@given(
    nodes_factor=st.lists(
        st.integers(min_value=1, max_value=10), min_size=3, max_size=3
    ),
    heads=st.integers(min_value=1, max_value=16),
    slice_num=st.integers(min_value=1, max_value=32),
    dropout=st.floats(min_value=0.0, max_value=1.0, exclude_max=True),
)
def test__can_pass_parameters_via_setting(
    nodes_factor: list[int],
    heads: int,
    slice_num: int,
    dropout: float,
):
    nodes = [x * heads for x in nodes_factor]
    setting = TransolverAttentionSetting(
        nodes=nodes,
        heads=heads,
        slice_num=slice_num,
        dropout=dropout,
    )
    model = TransolverAttention.from_setting(setting)

    assert model._nodes == nodes
    assert model._heads == heads
    assert model._slice_num == slice_num
    assert model._dropout_rate == dropout


@pytest.mark.parametrize(
    "nodes, heads, slice_num, dropout",
    [
        ([-1, 32, 5], 8, 32, 0.5),
        ([-1, 16, 20], 8, 8, 0.0),
    ],
)
@pytest.mark.parametrize(
    "input_shape, is_time_series",
    [
        ((10, 3, 1), False),
        ((20, 8), False),
        ((5, 20, 2, 1), False),
        ((2, 10, 3, 1), True),
        ((2, 10, 3, 5, 2, 16), True),
        ((1, 20, 6), True),
        ((5, 100, 5), True),
    ],
)
@pytest.mark.parametrize(
    "dimension", [{"Theta": 1}, {"M": 1, "T": -1}, None, {}]
)
def test__transolver_attention_for_vertexwise_tensor(
    nodes: list[int],
    heads: int,
    slice_num: int,
    dropout: float,
    input_shape: tuple[int],
    is_time_series: bool,
    dimension: dict | None,
):
    nodes[0] = input_shape[-1]
    _tensor = phlower_tensor(
        np.random.rand(*input_shape),
        is_time_series=is_time_series,
        dimension=dimension,
        dtype=torch.float32,
    )

    model = TransolverAttention(
        nodes=nodes,
        heads=heads,
        slice_num=slice_num,
        dropout=dropout,
    )

    result = model.forward(phlower_tensor_collection({"input": _tensor}))

    assert result.shape[:-1] == input_shape[:-1]
    assert result.shape[-1] == nodes[-1]
    if dimension is not None:
        assert (
            result.dimension.to_dict()
            == phlower_dimension_tensor(dimension).to_dict()
        )
    else:
        assert result.dimension is None


@pytest.mark.parametrize(
    "nodes, heads, slice_num, dropout",
    [
        ([-1, 32, 5], 8, 32, 0.5),
        ([-1, 16, 20], 8, 8, 0.0),
    ],
)
@pytest.mark.parametrize(
    "input_shapes, desired_shape",
    [([(10, 3, 8), (20, 3, 8)], (30, 3, -1))],
)
def test__transolver_attention_for_batched_tensor(
    nodes: list[int],
    heads: int,
    slice_num: int,
    dropout: float,
    input_shapes: list[tuple[int]],
    desired_shape: tuple[int],
):
    nodes[0] = input_shapes[0][-1]

    _tensors: list[PhlowerTensor] = [
        phlower_tensor(np.random.rand(*shape), dtype=torch.float32)
        for shape in input_shapes
    ]
    _tensor, batch_info = to_batch(_tensors)
    field = SimulationField({}, {"sample": batch_info})

    model = TransolverAttention(
        nodes=nodes,
        heads=heads,
        slice_num=slice_num,
        dropout=dropout,
        unbatch_key="sample",
    )

    result = model.forward(
        phlower_tensor_collection({"input": _tensor}), field_data=field
    )

    assert result.shape[:-1] == desired_shape[:-1]
    assert result.shape[-1] == nodes[-1]
