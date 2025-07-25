"""
This implementation refers to Pytorch Ignite EarlyStopping class.

See. https://pytorch.org/ignite/_modules/ignite/handlers/early_stopping.html#EarlyStopping
"""

from collections import OrderedDict
from collections.abc import Mapping
from typing import cast

from phlower.utils import get_logger
from phlower.utils.enums import (
    PhlowerHandlerRegisteredKey,
    PhlowerHandlerTrigger,
)
from phlower.utils.typing import AfterEvaluationOutput, IPhlowerHandler

__all__ = ["EarlyStopping"]


class EarlyStopping(IPhlowerHandler[AfterEvaluationOutput]):
    """EarlyStopping handler can be used to stop the training
      if no improvement after a given number of events.

    Args:
        patience: Number of events to wait if no improvement
          and then stop the training.
        min_delta: A minimum increase in the score to qualify as an improvement,
            i.e. an increase of less than or equal to `min_delta`,
              will count as no improvement.
        cumulative_delta: It True, `min_delta` defines an increase
          since the last `patience` reset, otherwise,
            it defines an increase after the last event. Default value is False.

    """

    _state_dict_all_req_keys = (
        "counter",
        "best_score",
    )

    @classmethod
    def name(cls) -> str:
        return "EarlyStopping"

    @classmethod
    def trigger(self) -> PhlowerHandlerTrigger:
        return PhlowerHandlerTrigger.epoch_completed

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        cumulative_delta: bool = False,
    ):
        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should be positive number.")

        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score: float | None = None
        self.logger = get_logger(__name__ + "." + self.__class__.__name__)

    def __call__(self, output: AfterEvaluationOutput) -> dict[str, bool]:
        if output.validation_eval_loss is None:
            self.logger.info(
                "Evaluation for validation dataset is missing. "
                "Evaluation for training dataset is used."
            )
            score = -1.0 * output.train_eval_loss
        else:
            score = -1.0 * output.validation_eval_loss

        if self.best_score is None:
            self.best_score = score

        elif score <= self.best_score + self.min_delta:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            self.logger.debug(
                f"EarlyStopping: {self.counter} / {self.patience}"
            )
            if self.counter >= self.patience:
                self.logger.info("EarlyStopping: Stop training")
                return {PhlowerHandlerRegisteredKey.TERMINATE: True}

        else:
            self.best_score = score
            self.counter = 0

        return {}

    def state_dict(self) -> OrderedDict[str, float]:
        """Method returns state dict with ``counter`` and ``best_score``.
        Can be used to save internal state of the class.
        """
        return OrderedDict(
            [
                ("counter", self.counter),
                ("best_score", cast(float, self.best_score)),
            ]
        )

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Method replace internal state of the class
          with provided state dict data.

        Args:
            state_dict: a dict with "counter" and "best_score" keys/values.
        """
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
