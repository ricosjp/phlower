import pytest

from phlower.utils.sliding_window import SlidingWindow


@pytest.mark.parametrize(
    "size, offset, stride, time_length, expected_total_items",
    [
        (2, 0, 1, 10, 9),
        (10, 0, 1, 9, 0),
        (2, 0, 2, 10, 5),
        (2, 1, 2, 5, 2),
        (2, 0, 2, 5, 2),
    ],
)
def test__sliding_window_get_max_items(
    size: int,
    offset: int,
    stride: int,
    time_length: int,
    expected_total_items: int,
):
    window = SlidingWindow(size=size, offset=offset, stride=stride)
    assert window.get_total_items(time_length) == expected_total_items


@pytest.mark.parametrize(
    "size, offset, stride, index, expected_slice",
    [
        (2, 0, 1, 0, slice(0, 2)),
        (2, 0, 1, 1, slice(1, 3)),
        (2, 0, 2, 2, slice(4, 6)),
        (2, 1, 2, 2, slice(5, 7)),
        (2, 0, 2, 3, slice(6, 8)),
        (2, 1, 5, 3, slice(16, 18)),
    ],
)
def test__sliding_window_get_slice(
    size: int,
    offset: int,
    stride: int,
    index: int,
    expected_slice: slice,
):
    window = SlidingWindow(size=size, offset=offset, stride=stride)
    desired = window.get_slice(index)
    assert desired.start == expected_slice.start
    assert desired.stop == expected_slice.stop
