from typing import Self

import pydantic

from phlower.utils.sliding_window import SlidingWindow


class SlidingWindowParameters(pydantic.RootModel):
    root: dict[str, SlidingWindow] | SlidingWindow

    def get_window(self, name: str) -> SlidingWindow:
        if isinstance(self.root, SlidingWindow):
            return self.root

        return self.root[name]

    def get_slice(self, name: str, index: int) -> slice:
        return self.get_window(name).get_slice(index)

    def get_unique_window_counts(
        self, name_to_time_length: dict[str, int]
    ) -> set[int]:
        return {
            self.get_window(name).get_total_items(time_length)
            for name, time_length in name_to_time_length.items()
        }


class SlidingWindowForStage(pydantic.BaseModel):
    is_active: bool = True
    """
    If True, sliding window is applied to time series data.
    """

    inputs: SlidingWindowParameters = pydantic.Field(
        default_factory=lambda: SlidingWindowParameters(
            SlidingWindow(offset=0, size=1, stride=1)
        )
    )
    """
    Sliding window settings for input data.
    """

    labels: SlidingWindowParameters = pydantic.Field(
        default_factory=lambda: SlidingWindowParameters(
            SlidingWindow(offset=0, size=1, stride=1)
        )
    )
    """
    Sliding window settings for label data.
    """

    treat_as_samples: bool = False
    """
    If True, every sliding window is one sample of the dataset.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    def windows_in_sequence(self) -> bool:
        return self.is_active and not self.treat_as_samples

    def windows_as_samples(self) -> bool:
        return self.is_active and self.treat_as_samples


class TimeSeriesSlidingSetting(pydantic.BaseModel):
    training_window_settings: SlidingWindowForStage = pydantic.Field(
        default_factory=lambda: SlidingWindowForStage(
            is_active=False,
            inputs=SlidingWindowParameters(
                root=SlidingWindow(offset=0, size=1, stride=1)
            ),
            labels=SlidingWindowParameters(
                root=SlidingWindow(offset=0, size=1, stride=1)
            ),
        )
    )
    """
    Settings for sliding window for training data.
    """

    validation_same_as_training: bool = False
    """
    If True, validation window settings are the same
    as training window settings.
    Defaults to False.
    """

    validation_window_settings: SlidingWindowForStage = pydantic.Field(
        default_factory=lambda: SlidingWindowForStage(
            is_active=False,
            inputs=SlidingWindowParameters(
                root=SlidingWindow(offset=0, size=1, stride=1)
            ),
            labels=SlidingWindowParameters(
                root=SlidingWindow(offset=0, size=1, stride=1)
            ),
        )
    )
    """
    Settings for sliding window for validation data.
    """

    @pydantic.model_validator(mode="after")
    def _check_same_as_windows(self) -> Self:
        if not self.validation_same_as_training:
            return self

        if self.validation_window_settings != self.training_window_settings:
            raise ValueError(
                "validation_window_settings must be the same as "
                "training_window_settings "
                "when validation_same_as_training is True."
            )
        return self

    @pydantic.model_validator(mode="before")
    @classmethod
    def _sync_validation_settings(cls, values: dict) -> dict:
        if not values.get("validation_same_as_training", False):
            return values

        if values.get("validation_window_settings", None) is not None:
            return values

        values["validation_window_settings"] = values.get(
            "training_window_settings", None
        )

        return values

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    def training_evaluation_window_settings(self) -> SlidingWindowForStage:
        # Training data whose windows are samples is already cut
        # into windows, so it is evaluated as it is.
        if self.training_window_settings.windows_as_samples():
            return self.training_window_settings
        return self.validation_window_settings
