from collections import OrderedDict
from collections.abc import Mapping

import numpy as np

from phlower.utils import get_logger
from phlower.utils.enums import (
    PhlowerHandlerTrigger,
)
from phlower.utils.typing import IPhlowerHandler, PhlowerHandlerAnalysisResult


class NaNStoppingHandler(IPhlowerHandler[float]):
    """NaNStoppingHandler handler raises PhlowerNaNDetectedError when loss
    has NaN values.
    """

    @classmethod
    def name(cls) -> str:
        return "NaNStoppingHandler"

    @classmethod
    def trigger(self) -> PhlowerHandlerTrigger:
        return PhlowerHandlerTrigger.iteration_completed

    def __init__(
        self,
    ):
        self.has_nan = False
        self.logger = get_logger(__name__ + "." + self.__class__.__name__)

    def __call__(self, output: float) -> PhlowerHandlerAnalysisResult:
        if np.isnan(output):
            self.has_nan = True
            return PhlowerHandlerAnalysisResult(
                terminate_training=True,
                reason="NaN detected in loss.",
            )
        return PhlowerHandlerAnalysisResult(
            terminate_training=False,
            reason=None,
        )

    def state_dict(self) -> OrderedDict[str, float]:
        """Method returns state dict with ``has_nan``.
        Can be used to save internal state of the class.
        """
        return OrderedDict(
            [
                ("has_nan", self.has_nan),
            ]
        )

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Method replace internal state of the class
          with provided state dict data.

        Args:
            state_dict: a dict with "has_nan" keys/values.
        """
        self.has_nan = state_dict["has_nan"]

    def update_activity(self, continue_count: int): ...
