import hypothesis.strategies as st
import pytest
from hypothesis import given
from phlower.services.trainer._loggings._record_io import LogRecordIO
from phlower.utils.typing import AfterEvaluationOutput


@pytest.mark.parametrize(
    "loss_keys, display_margin, desired",
    [
        (None, 1, "\nepoch train_loss validation_loss elapsed_time "),
        ([], 2, "\nepoch  train_loss  validation_loss  elapsed_time  "),
        (
            ["v", "p"],
            1,
            "\nepoch train_loss validation_loss details_tr/v "
            "details_tr/p details_val/v details_val/p elapsed_time ",
        ),
    ],
)
def test__headers(loss_keys: str | None, display_margin: int, desired: str):
    record_io = LogRecordIO(
        file_path="dummy", loss_keys=loss_keys, display_margin=display_margin
    )
    header = record_io.get_header()

    assert header == desired


@pytest.mark.parametrize(
    "loss_keys, display_margin",
    [
        (None, 5),
        (["v", "p"], 4),
    ],
)
@given(
    epoch=st.integers(min_value=0, max_value=100000),
    train_eval_loss=st.floats(width=64, allow_infinity=False, allow_nan=False),
    elapsed_time=st.floats(width=32, allow_infinity=False, allow_nan=False),
    validation_eval_loss=st.floats(
        width=64, allow_infinity=False, allow_nan=False
    ),
)
def test__to_str(
    loss_keys: str | None,
    display_margin: int,
    epoch: int,
    train_eval_loss: float,
    elapsed_time: float,
    validation_eval_loss: float,
):
    record_io = LogRecordIO(
        file_path="dummy", loss_keys=loss_keys, display_margin=display_margin
    )

    output = AfterEvaluationOutput(
        epoch=epoch,
        train_eval_loss=train_eval_loss,
        elapsed_time=elapsed_time,
        validation_eval_loss=validation_eval_loss,
        output_directory="dummy",
    )
    actual = record_io.to_str(output)
    assert len(actual.split()) == 4
