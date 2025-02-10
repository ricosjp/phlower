from typing import NamedTuple


class AfterEpochTrainingInfo(NamedTuple):
    """output data after training for one epoch"""

    epoch: int
    train_losses: list[float]


class AfterEvaluationOutput(NamedTuple):
    """output data after evaluation for one epoch"""

    train_eval_loss: float
    validation_eval_loss: float | None
