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
    [((5, 6, 7, 9, 10), "n d0 d1 d2 f"), ((5, 10), "n f"), ((5,), "n")],
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
        ((5, 6, 7, 9, 10), "t n d0 d1 f"),
        ((5, 10), "t n"),
        ((100, 5, 23), "t n f"),
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
        ((5, 6, 7, 9, 10), "x y z d0 f"),
        ((12, 6, 7, 9), "x y z f"),
        ((5, 6, 7, 9, 10, 78, 100), "x y z d0 d1 d2 f"),
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
        ((5, 6, 7, 9, 10), "t x y z f"),
        ((12, 6, 7, 9), "t x y z"),
        ((5, 6, 7, 9, 10, 78, 100), "t x y z d0 d1 f"),
    ],
)
def test__detect_time_series_voxel_pattern(shapes: tuple[int], desired: str):
    shape = PhlowerShapePattern(
        torch.Size(shapes), is_time_series=True, is_voxel=True
    )

    assert shape.get_pattern() == desired


# test for Rearrange


@pytest.mark.parametrize(
    "is_time_series, is_voxel",
    [(True, True), (False, False), (True, False), (False, True)],
)
@pytest.mark.parametrize(
    "orig_shape, to_shape, pattern",
    [
        (
            (10, 8, 8, 8, 3, 3, 16),
            (10, 8, 8, 8, 3, 3, 4, 4),
            "... (f g) -> ... f g",
        ),
        (
            (5, 3, 1, 2, 2, 2),
            (5, 3, 1, 2, 4),
            "... d0 d1 -> ... (d0 d1)",
        ),
        (
            (5, 3, 2, 2, 2),
            (5, 3, 2, 2, 2),
            "... f -> ... f",
        ),
    ],
)
def test__save_flags_of_shape_pattern_when_rearrange(
    is_time_series: bool,
    is_voxel: bool,
    orig_shape: tuple[int],
    to_shape: tuple[int],
    pattern: str,
):
    shape_pattern = PhlowerShapePattern(
        shape=orig_shape, is_time_series=is_time_series, is_voxel=is_voxel
    )

    actual = shape_pattern.rearrange(pattern, to_shape=to_shape)

    assert actual.shape == to_shape
    assert actual.is_time_series == is_time_series
    assert actual.is_voxel == is_voxel


@pytest.mark.parametrize(
    "orig_shape, is_time_series, is_voxel, to_shape, pattern, "
    "desired_time_series, desired_voxel",
    [
        (
            (5, 3, 1, 2, 2, 2),
            True,
            True,
            (5, 6, 4),
            "t x y z d0 d1 -> t (x y z) (d0 d1)",
            True,
            False,
        ),
        (
            (5, 3, 1, 2, 2, 2),
            True,
            True,
            (30, 4),
            "t x y z d0 d1 -> (t x y z) (d0 d1)",
            False,
            False,
        ),
        (
            (5, 3, 1, 2, 2, 2),
            True,
            True,
            (5, 3, 1, 2, 4),
            "t ... d0 d1 -> t ... (d0 d1)",
            True,
            True,
        ),
        (
            (4, 3, 2, 2, 2, 1),
            False,
            False,
            (4, 3, 2, 2, 2),
            "n ... d0 d1 -> n ... (d0 d1)",
            False,
            False,
        ),
    ],
)
def test__detech_shape_pattern_when_rearrange(
    orig_shape: tuple[int],
    is_time_series: bool,
    is_voxel: bool,
    to_shape: tuple[int],
    pattern: str,
    desired_time_series: bool,
    desired_voxel: bool,
):
    shape_pattern = PhlowerShapePattern(
        shape=orig_shape, is_time_series=is_time_series, is_voxel=is_voxel
    )

    actual = shape_pattern.rearrange(pattern, to_shape=to_shape)

    assert actual.shape == to_shape
    assert actual.is_time_series is desired_time_series
    assert actual.is_voxel is desired_voxel


@pytest.mark.parametrize(
    "shape, is_time_series, n_batch, desired",
    [
        ((1,), False, 1, True),
        ((2,), False, 2, True),
        ((2,), False, 1, False),
        ((4, 3, 16), False, 4, True),
        ((4, 3, 16), False, 2, False),
        ((10, 1), True, 1, True),
        ((10, 2), True, 2, True),
        ((10, 2), True, 1, False),
        ((10, 4, 3, 16), True, 4, True),
        ((10, 4, 3, 16), True, 2, False),
    ],
)
def test__is_global(
    shape: tuple[int],
    is_time_series: bool,
    n_batch: int,
    desired: bool,
):
    shape = PhlowerShapePattern(
        torch.Size(shape), is_time_series=is_time_series, is_voxel=False
    )

    assert shape.is_global(n_batch) == desired
