import pytest

from phlower.services.trainer._handler_functions import NaNStoppingHandler
from phlower.utils.enums import PhlowerHandlerTrigger
from phlower.utils.exceptions import PhlowerNaNDetectedError


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

    with pytest.raises(PhlowerNaNDetectedError):
        handler(float("nan"))
