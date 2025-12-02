import pathlib

import pytest
import torch
from phlower import phlower_tensor
from phlower._fields import ISimulationField
from phlower.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.nn import PhlowerGroupModule, PhlowerPresetGroupModuleAdapter
from phlower.settings import PhlowerSetting

DATA_DIR = pathlib.Path(__file__).parent / "data/preset_groups"


@pytest.mark.parametrize(
    "input_file, desired_name, desired_nn_name",
    [("simple_preset_group.yml", "Identity0", "Identity")],
)
def test__initialize_from_setting_class(
    input_file: str, desired_name: str, desired_nn_name: str
):
    setting = PhlowerSetting.read_yaml(DATA_DIR / input_file)

    setting.model.resolve()
    model = PhlowerGroupModule.from_setting(setting.model.network)

    preset = model._phlower_modules[0]
    assert isinstance(preset, PhlowerPresetGroupModuleAdapter)
    assert preset.name == desired_name
    assert preset.get_core_module().get_nn_name() == desired_nn_name


@pytest.mark.parametrize(
    "input_file, desired", [("simple_preset_group.yml", [])]
)
def test__destinations(input_file: str, desired: list[str]):
    setting = PhlowerSetting.read_yaml(DATA_DIR / input_file)

    setting.model.resolve()
    model = PhlowerGroupModule.from_setting(setting.model.network)

    preset = model._phlower_modules[0]
    assert preset.get_destinations() == desired


class MockModule(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self._phlower_modules = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features, 50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, out_features),
            ]
        )

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None = None,
    ) -> IPhlowerTensorCollections:
        x = data.unique_item()
        for module in self._phlower_modules:
            x = module(x)
        return phlower_tensor_collection({"out": x})


@pytest.mark.parametrize(
    "in_features, out_features",
    [
        (10, 5),
        (20, 15),
    ],
)
def test__forward(in_features: int, out_features: int):
    model = PhlowerPresetGroupModuleAdapter(
        layer=MockModule(in_features=in_features, out_features=out_features),
        name="MockModule",
        no_grad=False,
        input_keys=["in"],
        destinations=[],
        output_keys=["out"],
        input_nodes={"in": in_features},
        output_nodes={"out": out_features},
    )

    out = model.forward(
        data=phlower_tensor_collection(
            {"in": phlower_tensor(torch.randn(3, in_features))}
        ),
        field_data=None,
    )

    out_tensor = out.unique_item()
    assert out_tensor.to_tensor().requires_grad is True


@pytest.mark.parametrize(
    "in_features, out_features",
    [
        (10, 5),
        (20, 15),
    ],
)
def test__forward_with_no_grad(in_features: int, out_features: int):
    model = PhlowerPresetGroupModuleAdapter(
        layer=MockModule(in_features=in_features, out_features=out_features),
        name="MockModule",
        no_grad=True,
        input_keys=["in"],
        destinations=[],
        output_keys=["out"],
        input_nodes={"in": in_features},
        output_nodes={"out": out_features},
    )

    out = model.forward(
        data=phlower_tensor_collection(
            {"in": phlower_tensor(torch.randn(3, in_features))}
        ),
        field_data=None,
    )

    out_tensor = out.unique_item()
    assert out_tensor.to_tensor().requires_grad is False
