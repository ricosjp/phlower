import pytest
from phlower.services.trainer._handler_functions import EarlyStopping
from phlower.utils.enums import PhlowerHandlerTrigger
from phlower.utils.typing import AfterEvaluationOutput


def test__name():
    assert EarlyStopping.name() == "EarlyStopping"


def test__trigger():
    assert EarlyStopping.trigger() == PhlowerHandlerTrigger.epoch_completed


@pytest.mark.parametrize(
    "patience, min_delta, desired_msg",
    [
        (-10, 0.1, "Argument patience should be positive integer."),
        (10, -0.1, "Argument min_delta should be positive number."),
    ],
)
def test__cannot_initialize_invalid_args(
    patience: int, min_delta: float, desired_msg: str
):
    with pytest.raises(ValueError) as ex:
        _ = EarlyStopping(patience=patience, min_delta=min_delta)
    assert desired_msg in str(ex.value)


@pytest.mark.parametrize(
    "patience, min_delta, cumulative_delta, losses",
    [
        (3, 0.0, False, [0.2, 0.3, 0.4, 0.5]),
        (3, 0.1, False, [0.2, 0.1, 0.0, 0.0]),
        (3, 0.1, True, [0.2, 0.3, 0.4, 0.2]),
    ],
)
def test__early_stop_with_validation_loss(
    patience: int,
    min_delta: float,
    cumulative_delta: bool,
    losses: list[float],
):
    outputs = [
        AfterEvaluationOutput(
            epoch=i,
            train_eval_loss=0.0,
            validation_eval_loss=v,
            elapsed_time=10.0,
        )
        for i, v in enumerate(losses)
    ]

    handler = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        cumulative_delta=cumulative_delta,
    )

    results = []
    for output in outputs:
        results.append(handler(output))

    assert results[-1].get("TERMINATE")


@pytest.mark.parametrize(
    "patience, min_delta, cumulative_delta, losses",
    [
        (3, 0.0, False, [0.2, 0.3, 0.4, 0.5]),
        (3, 0.1, False, [0.2, 0.1, 0.0, 0.0]),
        (3, 0.1, True, [0.2, 0.3, 0.4, 0.2]),
    ],
)
def test__early_stop_with_training_loss(
    patience: int,
    min_delta: float,
    cumulative_delta: bool,
    losses: list[float],
):
    outputs = [
        AfterEvaluationOutput(
            epoch=i,
            train_eval_loss=v,
            validation_eval_loss=None,
            elapsed_time=10,
        )
        for i, v in enumerate(losses)
    ]

    handler = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        cumulative_delta=cumulative_delta,
    )

    results = []
    for output in outputs:
        results.append(handler(output))

    assert results[-1].get("TERMINATE")


@pytest.mark.parametrize(
    "patience, min_delta, cumulative_delta, losses, desired",
    [
        (10, 0.0, False, [0.2, 0.3, -0.4, 0.5, -0.1, -0.2], 0.4),
        (10, 0.2, False, [0.2, 0.1, 0.4, 0.1, 0.1, 0.2], -0.1),
        (10, 0.2, True, [0.2, 0.1, 0.4, 0.1, 0.1, 0.2], -0.2),
    ],
)
def test__calculate_best_score(
    patience: int,
    min_delta: float,
    cumulative_delta: bool,
    losses: list[float],
    desired: float,
):
    outputs = [
        AfterEvaluationOutput(
            epoch=i,
            train_eval_loss=v,
            validation_eval_loss=None,
            elapsed_time=10,
        )
        for i, v in enumerate(losses)
    ]

    handler = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        cumulative_delta=cumulative_delta,
    )

    for output in outputs:
        handler(output)

    assert handler.best_score == desired
