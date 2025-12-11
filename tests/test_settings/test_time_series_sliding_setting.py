import pytest
from phlower.settings._time_series_sliding_setting import (
    SlidingWindowForStage,
    SlidingWindowParameters,
    TimeSeriesSlidingSetting,
)
from phlower.utils.sliding_window import SlidingWindow


def test__default():
    setting = TimeSeriesSlidingSetting()
    assert setting.validation_same_as_training is False
    assert setting.training_window_settings.is_active is False
    assert setting.validation_window_settings.is_active is False


def test__set_validation_same_as_training():
    setting = TimeSeriesSlidingSetting(
        training_window_settings=SlidingWindowForStage(
            inputs=SlidingWindow(offset=0, size=2, stride=1),
            labels=SlidingWindow(offset=0, size=2, stride=1),
        ),
        validation_same_as_training=True,
    )

    assert (
        setting.validation_window_settings == setting.training_window_settings
    )


def test__raise_error_when_validation_same_as_training():
    with pytest.raises(
        ValueError,
        match=(
            "validation_window_settings must be the same "
            "as training_window_settings"
        ),
    ):
        TimeSeriesSlidingSetting(
            training_window_settings=SlidingWindowForStage(
                inputs=SlidingWindow(offset=0, size=2, stride=1),
                labels=SlidingWindow(offset=0, size=2, stride=1),
            ),
            validation_window_settings=SlidingWindowForStage(
                inputs=SlidingWindow(offset=0, size=3, stride=1),
                labels=SlidingWindow(offset=0, size=3, stride=1),
            ),
            validation_same_as_training=True,
        )


def test__sliding_window_parameters_when_single_item():
    setting = SlidingWindowParameters({"offset": 0, "size": 1, "stride": 1})

    assert setting.get_window("any_key").offset == 0
    assert setting.get_window("any_key").size == 1
    assert setting.get_window("any_key").stride == 1


def test__sliding_window_parameters_when_dict():
    setting = SlidingWindowParameters(
        {
            "key1": {"offset": 0, "size": 2, "stride": 1},
            "key2": {"offset": 1, "size": 3, "stride": 2},
        }
    )

    assert setting.get_window("key1").offset == 0
    assert setting.get_window("key1").size == 2
    assert setting.get_window("key1").stride == 1

    assert setting.get_window("key2").offset == 1
    assert setting.get_window("key2").size == 3
    assert setting.get_window("key2").stride == 2

    with pytest.raises(KeyError):
        setting.get_window("key3")


def test__reload_dumped_data_when_validation_same_as_training():
    setting = TimeSeriesSlidingSetting(
        training_window_settings=SlidingWindowForStage(
            inputs=SlidingWindow(offset=0, size=2, stride=1),
            labels=SlidingWindow(offset=0, size=2, stride=1),
        ),
        validation_same_as_training=True,
    )

    dumped = setting.model_dump()
    reloaded = TimeSeriesSlidingSetting.model_validate(dumped)

    assert setting == reloaded
