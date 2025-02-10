from typing import Any, NamedTuple


class AfterEpochInfo(NamedTuple):
    """output data after training for one epoch is finished"""

    epoch: int
    train_losses: list[float]


class AfterEpochRunnerOutput(NamedTuple):
    train_eval_loss: float
    validation_eval_loss: float
    user_defined: dict[str, Any] | None
