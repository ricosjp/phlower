import pytest
import torch
from phlower._base.tensors._tensor_shape import PhlowerShapePattern


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
