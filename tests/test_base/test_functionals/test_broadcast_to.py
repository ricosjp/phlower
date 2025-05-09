import pytest
import torch
from phlower import phlower_tensor
from phlower._base._functionals import broadcast_to
from phlower._base.tensors import PhlowerShapePattern


@pytest.mark.parametrize(
    "input_shape, target_shape, target_pattern, desired_shape",
    [
        ((10, 1), (10, 3, 3, 1), "n d1 d2 f", (10, 1, 1, 1)),
        ((10, 5), (10, 3, 5), "n d1 f", (10, 1, 5)),
        ((2,), (2, 1), "n f", (2, 1)),
        ((1, 2), (2, 2), "n f", (1, 2)),
    ],
)
def test__broadcast_to_correct_shape_for_standard_tensor(
    input_shape: tuple[int],
    target_pattern: tuple[int],
    target_shape: str,
    desired_shape: tuple[int],
):
    """Test that the broadcast_to function correctly broadcasts a tensor
    to the desired shape.
    """
    x = phlower_tensor(torch.rand(*input_shape))
    shape_pattern = PhlowerShapePattern.from_pattern(
        target_shape, target_pattern
    )
    result = broadcast_to(x, shape_pattern)
    assert tuple(result.shape) == desired_shape


@pytest.mark.parametrize(
    "input_shape, target_shape, target_pattern, desired_shape",
    [
        ((2, 10, 1), (10, 3, 3, 1), "n d1 d2 f", (2, 10, 1, 1, 1)),
        ((2, 10, 1), (2, 10, 3, 3, 1), "t n d1 d2 f", (2, 10, 1, 1, 1)),
        ((1, 10, 5), (10, 3, 5), "n d1 f", (1, 10, 1, 5)),
        ((4, 2), (2, 1), "n f", (4, 2, 1)),
        ((1, 1, 2), (1, 2, 2), "t n f", (1, 1, 2)),
    ],
)
def test__broadcast_to_correct_shape_for_time_series_tensor(
    input_shape: tuple[int],
    target_pattern: tuple[int],
    target_shape: str,
    desired_shape: tuple[int],
):
    """Test that the broadcast_to function correctly broadcasts a tensor
    to the desired shape.
    """
    x = phlower_tensor(torch.rand(*input_shape), is_time_series=True)
    shape_pattern = PhlowerShapePattern.from_pattern(
        target_shape, target_pattern
    )
    result = broadcast_to(x, shape_pattern)
    assert tuple(result.shape) == desired_shape
    assert result.is_time_series


@pytest.mark.parametrize(
    "input_shape, input_time_series, target_shape, "
    "target_pattern, desired_shape",
    [
        (
            (3, 3, 3, 2, 1),
            False,
            (3, 3, 3, 2, 1),
            "x y z d0 f",
            (3, 3, 3, 2, 1),
        ),
        (
            (1, 1, 1, 2, 1),
            False,
            (3, 3, 3, 2, 1),
            "x y z d1 f",
            (1, 1, 1, 2, 1),
        ),
        (
            (1, 1, 1, 2, 1),
            False,
            (3, 4, 5, 2, 2, 2, 1),
            "x y z d0 d1 d2 f",
            (1, 1, 1, 1, 1, 2, 1),
        ),
        (
            (5, 3, 3, 3, 2, 1),
            True,
            (3, 3, 3, 2, 1),
            "x y z d0 f",
            (5, 3, 3, 3, 2, 1),
        ),
        (
            (5, 3, 3, 3, 2, 1),
            True,
            (1, 3, 3, 3, 2, 1),
            "t x y z d0 f",
            (5, 3, 3, 3, 2, 1),
        ),
        (
            (1, 1, 1, 2, 1),
            True,
            (3, 4, 5, 2, 2, 2, 1),
            "t x y z d0 d1 f",
            (1, 1, 1, 2, 1, 1, 1),
        ),
    ],
)
def test__broadcast_to_correct_shape_for_voxel_tensor(
    input_shape: tuple[int],
    input_time_series: bool,
    target_pattern: tuple[int],
    target_shape: str,
    desired_shape: tuple[int],
):
    x = phlower_tensor(
        torch.rand(*input_shape),
        is_voxel=True,
        is_time_series=input_time_series,
    )
    shape_pattern = PhlowerShapePattern.from_pattern(
        target_shape, target_pattern
    )
    result = broadcast_to(x, shape_pattern)
    assert tuple(result.shape) == desired_shape
    assert result.is_voxel is True
    assert result.is_time_series == input_time_series


@pytest.mark.parametrize(
    "input_shape, input_time_series, target_shape, target_pattern",
    [
        ((10, 1), False, (2, 3, 4, 3, 3), "x y z d0 f"),
        ((1, 1, 1, 3), False, (2, 2, 2, 3), "x y z f"),
        ((2, 10, 1), True, (2, 3, 4, 3, 3), "t x y z f"),
    ],
)
def test__cannot_broadcast_to_voxel_shape_from_not_voxel(
    input_shape: tuple[int],
    input_time_series: bool,
    target_shape: tuple[int],
    target_pattern: str,
):
    x = phlower_tensor(
        torch.rand(*input_shape),
        is_voxel=False,
        is_time_series=input_time_series,
    )
    shape_pattern = PhlowerShapePattern.from_pattern(
        target_shape, target_pattern
    )

    with pytest.raises(ValueError) as ex:
        _ = broadcast_to(x, shape_pattern)
    assert "Cannot broadcast tensor with voxel shape" in str(ex.value)


@pytest.mark.parametrize(
    "input_shape, input_time_series, input_voxel, "
    "target_shape, target_pattern",
    [
        ((10, 3, 3, 3, 1), False, False, (10, 3, 3, 1), "n d1 d2 f"),
        ((10, 1, 1, 1, 1, 1), False, False, (10, 3, 3, 1), "n d1 d2 f"),
        ((2, 10, 3, 3, 3, 1), True, False, (10, 3, 3, 1), "n d1 d2 f"),
        ((1, 10, 1, 1, 1, 1, 1), True, False, (2, 10, 3, 3, 1), "t n d1 d2 f"),
        ((3, 3, 3, 2, 3, 1), False, True, (3, 3, 3, 2, 1), "x y z d0 f"),
        (
            (1, 1, 1, 2, 1, 1, 1, 1),
            True,
            True,
            (3, 4, 5, 2, 2, 2, 1),
            "t x y z d0 d1 f",
        ),
    ],
)
def test__cannot_broadcast_to_smaller_size_feature_shape(
    input_shape: tuple[int],
    input_time_series: bool,
    input_voxel: bool,
    target_shape: tuple[int],
    target_pattern: str,
):
    x = phlower_tensor(
        torch.rand(*input_shape),
        is_voxel=input_voxel,
        is_time_series=input_time_series,
    )
    shape_pattern = PhlowerShapePattern.from_pattern(
        target_shape, target_pattern
    )

    with pytest.raises(ValueError) as ex:
        _ = broadcast_to(x, shape_pattern)
    assert "Cannot broadcast tensor with feature shape" in str(ex.value)
