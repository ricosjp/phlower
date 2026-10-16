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


@pytest.mark.parametrize(
    "is_active, treat_as_samples, in_sequence, as_samples",
    [
        (True, False, True, False),
        (True, True, False, True),
        (False, False, False, False),
        (False, True, False, False),
    ],
)
def test__windows_in_sequence_or_as_samples(
    is_active: bool, treat_as_samples: bool, in_sequence: bool, as_samples: bool
):
    setting = SlidingWindowForStage(
        is_active=is_active, treat_as_samples=treat_as_samples
    )
    assert setting.windows_in_sequence() == in_sequence
    assert setting.windows_as_samples() == as_samples


@pytest.mark.parametrize(
    "is_active, treat_as_samples, desired_training",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
    ],
)
def test__training_evaluation_window_settings(
    is_active: bool, treat_as_samples: bool, desired_training: bool
):
    setting = TimeSeriesSlidingSetting(
        training_window_settings=SlidingWindowForStage(
            is_active=is_active,
            treat_as_samples=treat_as_samples,
            inputs=SlidingWindow(offset=0, size=2, stride=1),
            labels=SlidingWindow(offset=0, size=2, stride=1),
        ),
        validation_window_settings=SlidingWindowForStage(
            inputs=SlidingWindow(offset=0, size=3, stride=1),
            labels=SlidingWindow(offset=0, size=3, stride=1),
        ),
    )

    desired = (
        setting.training_window_settings
        if desired_training
        else setting.validation_window_settings
    )
    assert setting.training_evaluation_window_settings() == desired


@pytest.mark.parametrize(
    "parameters, name, index, desired",
    [
        ({"offset": 0, "size": 2, "stride": 1}, "u", 3, slice(3, 5)),
        (
            {
                "u": {"offset": 0, "size": 2, "stride": 1},
                "p": {"offset": 1, "size": 1, "stride": 2},
            },
            "p",
            3,
            slice(7, 8),
        ),
    ],
)
def test__get_slice(parameters: dict, name: str, index: int, desired: slice):
    window_parameters = SlidingWindowParameters.model_validate(parameters)
    assert window_parameters.get_slice(name, index) == desired


@pytest.mark.parametrize(
    "name_to_time_length, desired",
    [
        ({"u": 5, "p": 8}, {4}),
        ({"u": 5, "p": 6}, {4, 3}),
        ({"u": 1}, {0}),
        ({}, set()),
    ],
)
def test__get_unique_window_counts(
    name_to_time_length: dict[str, int], desired: set
):
    window_parameters = SlidingWindowParameters.model_validate(
        {
            "u": {"offset": 0, "size": 2, "stride": 1},
            "p": {"offset": 1, "size": 1, "stride": 2},
        }
    )
    assert (
        window_parameters.get_unique_window_counts(name_to_time_length)
        == desired
    )
