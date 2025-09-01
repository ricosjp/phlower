import pathlib

import pytest
import yaml
from phlower.settings import PhlowerModelSetting
from phlower.utils.enums import PhysicalDimensionSymbolType

# region E2E tests only for EnEquivariantMLPSettings

_TEST_DATA_DIR = (
    pathlib.Path(__file__).parent / "data/similarity_equivariant_mlp_setting"
)


@pytest.mark.parametrize("yaml_file", ["correct_scale_names.yml"])
def test__correct_scale_names(yaml_file: str):
    with open(_TEST_DATA_DIR / yaml_file) as fr:
        content = yaml.load(fr, Loader=yaml.SafeLoader)

    setting = PhlowerModelSetting(**content["model"])
    fields = setting.fields
    scale_names = setting.network.modules[0].nn_parameters.scale_names
    for field in fields:
        assert field.name in scale_names.values()
    for scale_key in scale_names.keys():
        assert PhysicalDimensionSymbolType.is_exist(scale_key)


@pytest.mark.parametrize("yaml_file", ["incorrect_scale_names.yml"])
def test__incorrect_scale_names(yaml_file: str):
    with open(_TEST_DATA_DIR / yaml_file) as fr:
        content = yaml.load(fr, Loader=yaml.SafeLoader)

    with pytest.raises(ValueError):
        PhlowerModelSetting(**content["model"])


# endregion
