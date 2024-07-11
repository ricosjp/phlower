import pydantic
import pytest

from phlower.settings._scaling_setting import ScalerInputParameters


def test__must_be_true():
    with pytest.raises(pydantic.ValidationError):
        _ = ScalerInputParameters(method="identity", join_fitting=False)
