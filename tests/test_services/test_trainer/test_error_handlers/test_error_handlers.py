import pathlib

import pytest

from phlower.services.trainer._error_handlers import (
    ErrorHandlerChain,
    IPhlowerErrorHandler,
    PhlowerErrorHandlerFactory,
)
from phlower.services.utils import TrainingTerminatedState
from phlower.settings import PhlowerSetting

# region Dump traceback


def test__dump_traceback(tmp_path: pathlib.Path):
    handler = ErrorHandlerChain(["dump_traceback"])
    setting = PhlowerSetting()

    state = TrainingTerminatedState(
        output_directory=tmp_path,
        setting=setting,
        exception=RuntimeError("Test exception"),
        traceback_info="Traceback info",
    )
    assert handler.should_suppress(state) is None

    assert (tmp_path / "training_error.txt").exists()
    content = (tmp_path / "training_error.txt").read_text()
    assert "Test exception" in content


# endregion


# region Allow when one epoch completed


@pytest.mark.parametrize(
    "n_pth_files",
    [1, 4, 10],
)
@pytest.mark.parametrize("ext", [".pth", ".pth.enc"])
def test__allow_when_one_epoch_completed(
    n_pth_files: int, ext: str, tmp_path: pathlib.Path
):
    handler = ErrorHandlerChain(["allow_when_one_epoch_completed"])
    setting = PhlowerSetting()

    # Create a .pth file to simulate one epoch completed
    (tmp_path / "weights").mkdir()
    for i in range(n_pth_files):
        (tmp_path / "weights" / f"snapshot_epoch_{i}{ext}").write_text(
            "dummy model"
        )

    state = TrainingTerminatedState(
        output_directory=tmp_path,
        setting=setting,
        exception=RuntimeError("Test exception"),
        traceback_info="Traceback info",
    )
    assert handler.should_suppress(state) is True


@pytest.mark.parametrize(
    "error_type, msg",
    [
        (RuntimeError, "Test exception"),
        (ValueError, "Test value error"),
        (KeyError, "Test key error"),
    ],
)
def test__transfer_error_when_no_pth_files(
    error_type: type[Exception], msg: str, tmp_path: pathlib.Path
):
    handler = ErrorHandlerChain(["allow_when_one_epoch_completed"])
    setting = PhlowerSetting()

    (tmp_path / "weights").mkdir()
    state = TrainingTerminatedState(
        output_directory=tmp_path,
        setting=setting,
        exception=error_type(msg),
        traceback_info="Traceback info",
    )
    assert handler.should_suppress(state) is None


# endregion


# register Error


class TestErrorHandler(IPhlowerErrorHandler):
    def should_suppress(
        self, terminated_state: TrainingTerminatedState
    ) -> None:
        pass


def test__register_error_handler():
    with pytest.raises(NotImplementedError, match="is not registered"):
        _ = ErrorHandlerChain(["test_handler"])

    PhlowerErrorHandlerFactory.register("test_handler", TestErrorHandler)
    _ = ErrorHandlerChain(["test_handler"])

    with pytest.raises(ValueError, match="already existed"):
        PhlowerErrorHandlerFactory.register("test_handler", TestErrorHandler)

    PhlowerErrorHandlerFactory.register(
        "test_handler", TestErrorHandler, overwrite=True
    )


# endregion
