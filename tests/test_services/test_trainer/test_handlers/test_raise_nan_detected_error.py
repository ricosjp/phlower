
import pytest

from phlower.services.trainer._handler_functions import RaiseNaNDetectedError
from phlower.utils.enums import PhlowerHandlerTrigger
from phlower.utils.exceptions import PhlowerNaNDetectedError


def test__name():
    assert RaiseNaNDetectedError.name() == "RaiseNaNDetectedError"


def test__trigger():
    assert RaiseNaNDetectedError.trigger() == PhlowerHandlerTrigger.iteration


def test__raise_when_nan():
    handler = RaiseNaNDetectedError()

    handler(0.1)

    with pytest.raises(PhlowerNaNDetectedError):
        handler(float("nan"))
