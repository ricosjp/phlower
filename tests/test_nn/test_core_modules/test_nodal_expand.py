import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from phlower_tensor import (
    SimulationField,
    phlower_dimension_tensor,
    phlower_tensor,
)
from phlower_tensor.collections import phlower_tensor_collection
from phlower_tensor.functionals import to_batch

from phlower.nn import NodalExpand, Pooling
from phlower.settings._module_settings import NodalExpandSetting


def test__from_setting():
    model = NodalExpand.from_setting(
        NodalExpandSetting(
            reference_key="reference", nodes=[3, 3], unbatch_key="mesh"
        )
    )
    assert model._reference_key == "reference"
    assert model._nodes == [3, 3]
    assert model._unbatch_key == "mesh"
    assert list(model.parameters()) == []


@pytest.mark.parametrize(
    "input_shape, reference_shape, reference_time_series, desired_shape",
    [
        ((1, 5), (10, 3), False, (10, 5)),
        ((1, 2, 3), (4, 2, 3), False, (4, 2, 3)),
        ((1, 2, 3), (4, 2, 3), True, (2, 2, 3)),
    ],
)
@pytest.mark.parametrize("dimension", [None, {}, {"L": 1, "T": -1}])
def test__nodal_expand_forward(
    input_shape: tuple[int, ...],
    reference_shape: tuple[int, ...],
    reference_time_series: bool,
    desired_shape: tuple[int, ...],
    dimension: dict | None,
):
    source = phlower_tensor(torch.randn(input_shape), dimension=dimension)
    reference = phlower_tensor(
        torch.randn(reference_shape), is_time_series=reference_time_series
    )
    actual = NodalExpand(reference_key="reference")(
        phlower_tensor_collection({"source": source, "reference": reference})
    )
    torch.testing.assert_close(actual.to_tensor().shape, desired_shape)

    assert actual.is_time_series is False
    if dimension is None:
        assert actual.dimension is None
    else:
        assert actual.dimension == phlower_dimension_tensor(dimension)


@pytest.mark.parametrize(
    "input_shape, reference_shape, reference_time_series, desired_shape",
    [
        ((7, 1, 5), (10, 3), False, (7, 10, 5)),
        ((10, 1, 2, 3), (4, 2, 3), False, (10, 4, 2, 3)),
        ((10, 1, 2, 3), (4, 2, 3), True, (10, 2, 2, 3)),
        ((10, 1, 2, 3), (10, 2, 3), False, (10, 10, 2, 3)),
    ],
)
@pytest.mark.parametrize("dimension", [None, {}, {"L": 1, "T": -1}])
def test__nodal_expand_forward_w_time_series(
    input_shape: tuple[int, ...],
    reference_shape: tuple[int, ...],
    reference_time_series: bool,
    desired_shape: tuple[int, ...],
    dimension: dict | None,
):
    source = phlower_tensor(
        torch.randn(input_shape), is_time_series=True, dimension=dimension
    )
    reference = phlower_tensor(
        torch.randn(reference_shape), is_time_series=reference_time_series
    )
    actual = NodalExpand(reference_key="reference")(
        phlower_tensor_collection({"source": source, "reference": reference})
    )
    torch.testing.assert_close(actual.to_tensor().shape, desired_shape)
    assert actual.is_time_series is True
    if dimension is None:
        assert actual.dimension is None
    else:
        assert actual.dimension == phlower_dimension_tensor(dimension)


@pytest.mark.parametrize(
    "input_shape, reference_shape, reference_time_series, desired_shape",
    [
        ((1, 5), (10, 3), False, (10, 5)),
        ((1, 2, 3), (4, 2, 3), False, (4, 2, 3)),
        ((1, 2, 3), (4, 2, 3), True, (2, 2, 3)),
    ],
)
@given(reference_first=st.booleans())
def test__reference_is_selected_by_name(
    reference_first: bool,
    input_shape: tuple[int, ...],
    reference_shape: tuple[int, ...],
    reference_time_series: bool,
    desired_shape: tuple[int, ...],
):

    source = phlower_tensor(torch.randn(input_shape))
    reference = phlower_tensor(
        torch.rand(reference_shape), is_time_series=reference_time_series
    )
    items = [("features", source), ("mesh", reference)]
    if reference_first:
        items.reverse()

    actual = NodalExpand(reference_key="mesh")(
        phlower_tensor_collection(dict(items))
    )

    torch.testing.assert_close(actual.to_tensor().shape, desired_shape)


@pytest.mark.parametrize(
    "input_shapes, is_time_series",
    [([(2, 3, 3, 2), (2, 5, 3, 2)], True), ([(3, 2), (5, 2)], False)],
)
@pytest.mark.parametrize("pool_operator_name", ["mean", "max"])
def test__batched_pooling_then_expand(
    input_shapes: list[tuple[int, ...]],
    is_time_series: bool,
    pool_operator_name: str,
):
    samples = [
        phlower_tensor(
            torch.randn(shape),
            is_time_series=is_time_series,
            dimension={"L": 1},
        )
        for shape in input_shapes
    ]
    reference, batch_info = to_batch(samples, 1 if is_time_series else 0)
    field = SimulationField({}, {"mesh": batch_info})
    data = phlower_tensor_collection({"input": reference})
    pooled = Pooling(pool_operator_name=pool_operator_name, unbatch_key="mesh")(
        data, field_data=field
    )
    actual = NodalExpand(reference_key="reference", unbatch_key="mesh")(
        phlower_tensor_collection({"source": pooled, "reference": reference}),
        field_data=field,
    )
    expected = Pooling(
        pool_operator_name=pool_operator_name,
        unbatch_key="mesh",
        expand_after_pooling=True,
    )(data, field_data=field)
    torch.testing.assert_close(actual.to_tensor(), expected.to_tensor())
    assert actual.dimension == reference.dimension
    assert actual.is_time_series == is_time_series


# region tests for error handling in NodalExpand


def test__reject_missing_reference_key():
    data = phlower_tensor_collection(
        {
            "source": phlower_tensor(torch.ones(1, 3)),
            "reference": phlower_tensor(torch.ones(5, 7)),
        }
    )
    with pytest.raises(ValueError, match="reference_key 'missing'"):
        NodalExpand(reference_key="missing")(data)


@pytest.mark.parametrize("n_inputs", [0, 1, 3])
def test__reject_invalid_input_count(n_inputs: int):
    data = phlower_tensor_collection(
        {str(i): phlower_tensor(torch.ones(1, 2)) for i in range(n_inputs)}
    )
    with pytest.raises(ValueError, match="Exactly two inputs"):
        NodalExpand(reference_key="reference")(data)


def test__reject_non_expandable_node_count():
    data = phlower_tensor_collection(
        {
            "source": phlower_tensor(torch.ones(2, 3)),
            "reference": phlower_tensor(torch.ones(5, 7)),
        }
    )
    with pytest.raises(
        ValueError,
        match="NodalExpand requires the source to have exactly one node",
    ):
        NodalExpand(reference_key="reference")(data)


def test__unbatch_requires_field_data():
    data = phlower_tensor_collection(
        {
            "source": phlower_tensor(torch.ones(1, 3)),
            "reference": phlower_tensor(torch.ones(5, 7)),
        }
    )
    with pytest.raises(ValueError, match="field data is not provided"):
        NodalExpand(reference_key="reference", unbatch_key="mesh")(data)


@pytest.mark.parametrize("voxel_key", ["source", "reference"])
def test__reject_voxel_tensors(voxel_key: str):
    data = {
        "source": phlower_tensor(torch.ones(1, 3)),
        "reference": phlower_tensor(torch.ones(5, 7)),
    }
    data[voxel_key] = phlower_tensor(torch.ones(1, 2, 3, 4), is_voxel=True)
    with pytest.raises(ValueError, match="voxel tensors"):
        NodalExpand(reference_key="reference")(phlower_tensor_collection(data))


# endregion
