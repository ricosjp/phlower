from collections import OrderedDict
from collections.abc import Mapping

import numpy as np

from phlower.utils import get_logger
from phlower.utils.enums import (
    PhlowerHandlerTrigger,
)
from phlower.utils.exceptions import PhlowerNaNDetectedError
from phlower.utils.typing import IPhlowerHandler


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

    def __call__(self, output: float) -> dict[str, bool]:
        if np.isnan(output):
            self.has_nan = True
            raise PhlowerNaNDetectedError

        return {}

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
