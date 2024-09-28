import pytest
import torch
from phlower._base.tensors._tensor_shape import PhlowerShapePattern
from phlower.utils.exceptions import PhlowerIncompatibleTensorError


@pytest.mark.parametrize(
    "shape, pattern, desired_time_series, desired_voxel",
    [
        ((3, 4, 5, 3), "t x y z", True, True),
        ((3, 4, 5, 3), "n...", False, False),
        ((3, 4, 5, 3), "t n ...", True, False),
        ((3, 4, 5, 3), "x y z ...", False, True),
        ((10, 2, 5, 3), "(n)...", False, False),
        ((3, 4, 5, 3), "(t) (x) (  y) (z )", True, True),
        ((3, 4, 5), "t (x y z) f", True, False),
        ((3, 4, 5, 3), "n a100 a2 f", False, False),
    ],
)
def test__detect_flag_when_using_from_pattern(
    shape: tuple[int],
    pattern: str,
    desired_time_series: bool,
    desired_voxel: bool,
):
    shape_pattern = PhlowerShapePattern.from_pattern(shape, pattern)

    assert shape_pattern.is_time_series == desired_time_series
    assert shape_pattern.is_voxel == desired_voxel


@pytest.mark.parametrize(
    "shape, pattern",
    [
        ((3, 4, 5), "(x y z)"),
        ((3, 4, 5), "x y "),
        ((10, 3, 4, 5, 9), "(x y z) d f"),
        ((3, 4, 5), "t (x y z)"),
    ],
)
def test__raise_error_for_invalid_pattern(shape: tuple[int], pattern: str):
    with pytest.raises(PhlowerIncompatibleTensorError):
        _ = PhlowerShapePattern.from_pattern(shape, pattern)


@pytest.mark.parametrize(
    "shapes, desired",
    [((5, 6, 7, 9, 10), "n a0 a1 a2 a3"), ((5, 10), "n a0"), ((5,), "n")],
)
def test__detect_not_time_series_vertexwise_pattern(
    shapes: tuple[int], desired: str
):
    shape = PhlowerShapePattern(
        torch.Size(shapes), is_time_series=False, is_voxel=False
    )

    assert shape.get_pattern() == desired


@pytest.mark.parametrize(
    "shapes, desired",
    [
        ((5, 6, 7, 9, 10), "t n a0 a1 a2"),
        ((5, 10), "t n"),
        ((100, 5, 23), "t n a0"),
    ],
)
def test__detect_time_series_vertexwise_pattern(
    shapes: tuple[int], desired: str
):
    shape = PhlowerShapePattern(
        torch.Size(shapes), is_time_series=True, is_voxel=False
    )

    assert shape.get_pattern() == desired


@pytest.mark.parametrize(
    "shapes, desired",
    [
        ((5, 6, 7, 9, 10), "x y z a0 a1"),
        ((12, 6, 7, 9), "x y z a0"),
        ((5, 6, 7, 9, 10, 78, 100), "x y z a0 a1 a2 a3"),
    ],
)
def test__detect_not_time_series_voxel_pattern(
    shapes: tuple[int], desired: str
):
    shape = PhlowerShapePattern(
        torch.Size(shapes), is_time_series=False, is_voxel=True
    )

    assert shape.get_pattern() == desired


@pytest.mark.parametrize(
    "shapes, desired",
    [
        ((5, 6, 7, 9, 10), "t x y z a0"),
        ((12, 6, 7, 9), "t x y z"),
        ((5, 6, 7, 9, 10, 78, 100), "t x y z a0 a1 a2"),
    ],
)
def test__detect_time_series_voxel_pattern(shapes: tuple[int], desired: str):
    shape = PhlowerShapePattern(
        torch.Size(shapes), is_time_series=True, is_voxel=True
    )

    assert shape.get_pattern() == desired
