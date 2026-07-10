from phlower.services.trainer._handler_functions import NaNStoppingHandler
from phlower.utils.enums import PhlowerHandlerTrigger


def test__name():
    assert NaNStoppingHandler.name() == "NaNStoppingHandler"


def test__trigger():
    assert (
        NaNStoppingHandler.trigger()
        == PhlowerHandlerTrigger.iteration_completed
    )


def test__raise_when_nan():
    handler = NaNStoppingHandler()

    handler(0.1)

    result = handler(float("nan"))
    assert result.terminate_training
