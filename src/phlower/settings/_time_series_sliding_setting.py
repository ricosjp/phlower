import pydantic

from phlower.utils.sliding_window import SlidingWindow


class SlidingWindowParameters(pydantic.RootModel):
    root: dict[str, SlidingWindow] | SlidingWindow

    def get_window(self, name: str) -> SlidingWindow:
        if isinstance(self.root, SlidingWindow):
            return self.root

        return self.root[name]


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

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


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

    @pydantic.model_validator(mode="before")
    @classmethod
    def _sync_validation_settings(cls, values: dict) -> dict:
        if not values.get("validation_same_as_training", False):
            return values

        if values.get("validation_window_settings", None) is not None:
            raise ValueError(
                "validation_window_settings must be None "
                "when validation_same_as_training is True."
            )

        values["validation_window_settings"] = values.get(
            "training_window_settings", None
        )

        return values

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)
