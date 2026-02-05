from __future__ import annotations

import pathlib

import pytest
import torch
from phlower_tensor import SimulationField
from phlower_tensor.collections.tensors import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn import Concatenator, PhlowerGroupModule
from phlower.nn._core_modules import _name2model
from phlower.nn._utils import (
    PhlowerRunTimeError,
)
from phlower.settings import PhlowerSetting
from phlower.settings._module_parameter_setting import _name_to_setting
from phlower.settings._module_settings import ConcatenatorSetting


class ErrorConcatenator(Concatenator):
    @classmethod
    def from_setting(cls, setting: ConcatenatorSetting) -> ErrorConcatenator:
        return ErrorConcatenator(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        return "ErrorConcatenator"

    def name(self) -> str:
        return "ErrorConcatenator"

    def forward(self, data: IPhlowerTensorCollections, **kwards):
        raise ValueError("Intentional error for testing")


class ErrorContenatorSetting(ConcatenatorSetting):
    @classmethod
    def get_nn_type(cls) -> str:
        return "ErrorConcatenator"


@pytest.fixture
def set_test_module():
    _name_to_setting.update({"ErrorConcatenator": ErrorContenatorSetting})
    _name2model.update({"ErrorConcatenator": ErrorConcatenator})
    yield
    _name2model.pop("ErrorConcatenator")
    _name_to_setting.pop("ErrorConcatenator")


@pytest.fixture
def dummy_inputs() -> IPhlowerTensorCollections:
    inputs = {
        "feature0": torch.rand(100, 10),
        "feature1": torch.rand(100, 12),
        "support0": torch.rand(100, 100),
    }
    return phlower_tensor_collection(inputs)


@pytest.fixture
def dummy_field() -> SimulationField:
    inputs = {
        "support1": torch.rand(100, 100),
    }

    return SimulationField(field_tensors=phlower_tensor_collection(inputs))


def test_attach_location_to_error_message(
    set_test_module: None,
    dummy_inputs: IPhlowerTensorCollections,
    dummy_field: SimulationField,
):
    yaml_dir = pathlib.Path(__file__).parent / "data/group/error_test.yml"
    setting = PhlowerSetting.read_yaml(yaml_dir)
    setting.model.resolve()

    model = PhlowerGroupModule.from_setting(setting.model.network)

    with pytest.raises(PhlowerRunTimeError) as exc_info:
        _ = model(dummy_inputs, field_data=dummy_field)

    assert "CYCLE_MODEL -> SUB_GROUP -> CONCAT" in str(exc_info.value)
    assert "Intentional error for testing" in str(exc_info.value)
