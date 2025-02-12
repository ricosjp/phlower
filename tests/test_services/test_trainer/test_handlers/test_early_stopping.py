import pytest
from phlower.services.trainer._handler_functions import EarlyStopping
from phlower.services.trainer._pass_items import AfterEvaluationOutput


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
        AfterEvaluationOutput(train_eval_loss=0.0, validation_eval_loss=v)
        for v in losses
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
        AfterEvaluationOutput(train_eval_loss=v, validation_eval_loss=None)
        for v in losses
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
        AfterEvaluationOutput(train_eval_loss=v, validation_eval_loss=None)
        for v in losses
    ]

    handler = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        cumulative_delta=cumulative_delta,
    )

    for output in outputs:
        handler(output)

    assert handler.best_score == desired
