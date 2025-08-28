import numpy as np
import pytest
import torch
from phlower import phlower_tensor
from phlower.collections import phlower_tensor_collection
from phlower.data import LumpedTensorData
from phlower.services.trainer._sliding_window_helper import SlidingWindowHelper
from phlower.settings._time_series_sliding_setting import (
    TimeSeriesSlidingSetting,
)


@pytest.fixture
def lumped_data_sample() -> LumpedTensorData:
    return LumpedTensorData(
        x_data=phlower_tensor_collection({"a": torch.rand(3, 10, 1)}),
        field_data=phlower_tensor_collection({"field": torch.rand(3, 5, 1)}),
        y_data=phlower_tensor_collection({"a": torch.rand(3, 10, 1)}),
    )


def assert_equal_tensor_collections(
    actual: LumpedTensorData,
    expected: LumpedTensorData,
) -> None:
    assert len(actual.x_data) == len(expected.x_data)
    for k in actual.x_data.keys():
        np.testing.assert_array_almost_equal(
            actual.x_data[k].numpy(), expected.x_data[k].numpy()
        )

    assert len(actual.y_data) == len(expected.y_data)
    for k in actual.y_data.keys():
        np.testing.assert_array_almost_equal(
            actual.y_data[k].numpy(), expected.y_data[k].numpy()
        )
    assert len(actual.field_data.keys()) == len(expected.field_data.keys())
    for k in actual.field_data.keys():
        np.testing.assert_array_almost_equal(
            actual.field_data[k].numpy(), expected.field_data[k].numpy()
        )


def test__sliding_window_helper_no_sliding_window(
    lumped_data_sample: LumpedTensorData,
):
    setting = TimeSeriesSlidingSetting()  # default is no sliding window
    helper = SlidingWindowHelper(
        lumped_data=lumped_data_sample,
        sliding_setting=setting.training_window_settings,
    )

    assert len(helper) == 1
    actual = list(helper)[0]

    assert_equal_tensor_collections(actual, lumped_data_sample)


def create_lumped_time_series_data_sample(
    input_time_series: int, label_time_series: bool
) -> LumpedTensorData:
    return LumpedTensorData(
        x_data=phlower_tensor_collection(
            {
                "a": phlower_tensor(
                    torch.rand(input_time_series, 10, 1), is_time_series=True
                ),
                "b": phlower_tensor(torch.rand(1, 5, 1), is_time_series=False),
            }
        ),
        field_data=phlower_tensor_collection({"field": torch.rand(1, 1)}),
        y_data=phlower_tensor_collection(
            {
                "a": phlower_tensor(
                    torch.rand(label_time_series, 10, 1), is_time_series=True
                ),
                "b": phlower_tensor(torch.rand(1, 5, 1), is_time_series=False),
            }
        ),
    )


@pytest.mark.parametrize(
    "offset, size, stride, expected_total_size",
    [(0, 2, 2, 5), (0, 3, 2, 4), (1, 2, 2, 4)],
)
def test__check_total_size_is_matched(
    offset: int,
    size: int,
    stride: int,
    expected_total_size: int,
):
    lumped_data_sample = create_lumped_time_series_data_sample(10, 10)
    setting = TimeSeriesSlidingSetting(
        training_window_settings={
            "inputs": {"offset": offset, "size": size, "stride": stride},
            "labels": {"offset": offset, "size": size, "stride": stride},
        },
    )
    helper = SlidingWindowHelper(
        lumped_data=lumped_data_sample,
        sliding_setting=setting.training_window_settings,
    )

    assert len(helper) == expected_total_size


def test__check_total_size_is_matched_for_different_setting():
    lumped_data_sample = create_lumped_time_series_data_sample(10, 30)
    setting = TimeSeriesSlidingSetting(
        training_window_settings={
            "inputs": {"offset": 0, "size": 5, "stride": 2},
            "labels": {"offset": 10, "size": 10, "stride": 4},
        },
    )
    helper = SlidingWindowHelper(
        lumped_data=lumped_data_sample,
        sliding_setting=setting.training_window_settings,
    )

    assert len(helper) == 3


def test__raise_error_when_inconsistent_length():
    lumped_data_sample = create_lumped_time_series_data_sample(10, 5)
    setting = TimeSeriesSlidingSetting(
        training_window_settings={
            "inputs": {"offset": 0, "size": 2, "stride": 2},
            "labels": {"offset": 0, "size": 2, "stride": 2},
        },
    )

    with pytest.raises(ValueError, match="Inconsistent lengths after applying"):
        SlidingWindowHelper(
            lumped_data=lumped_data_sample,
            sliding_setting=setting.training_window_settings,
        )


@pytest.mark.parametrize(
    "offset, size, stride",
    [(0, 2, 2), (0, 3, 2), (1, 2, 2)],
)
def test__iterate_sliding_window(
    offset: int,
    size: int,
    stride: int,
):
    lumped_data_sample = create_lumped_time_series_data_sample(10, 10)
    setting = TimeSeriesSlidingSetting(
        training_window_settings={
            "inputs": {"offset": offset, "size": size, "stride": stride},
            "labels": {"offset": offset, "size": size, "stride": stride},
        },
    )
    helper = SlidingWindowHelper(
        lumped_data=lumped_data_sample,
        sliding_setting=setting.training_window_settings,
    )

    for i, item in enumerate(helper):
        start = i * stride + offset
        end = start + size

        for k, v in item.x_data.items():
            if v.is_time_series:
                np.testing.assert_array_almost_equal(
                    v.numpy(),
                    lumped_data_sample.x_data[k][start:end, :].numpy(),
                )
            else:
                np.testing.assert_array_almost_equal(
                    v.numpy(), lumped_data_sample.x_data[k].numpy()
                )

        for k, v in item.y_data.items():
            if v.is_time_series:
                np.testing.assert_array_almost_equal(
                    v.numpy(),
                    lumped_data_sample.y_data[k][start:end, :].numpy(),
                )
            else:
                np.testing.assert_array_almost_equal(
                    v.numpy(), lumped_data_sample.y_data[k].numpy()
                )
        for k, v in item.field_data.items():
            np.testing.assert_array_almost_equal(
                v.numpy(), lumped_data_sample.field_data[k].numpy()
            )
