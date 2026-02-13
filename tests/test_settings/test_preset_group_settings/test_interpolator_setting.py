import pathlib

import pytest
import yaml

from phlower.settings import PhlowerModelSetting

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/interpolator_settings"


@pytest.mark.parametrize("yaml_file", ["interpolator.yml"])
def test__parameters_correct(yaml_file: str):
    with open(_TEST_DATA_DIR / yaml_file) as fr:
        content = yaml.load(fr, Loader=yaml.SafeLoader)

    setting = PhlowerModelSetting(**content["model"])
    setting.network.resolve(is_first=True)

    assert len(content["misc"]["tests"].items()) > 0

    interpolator = setting.network.search_module_setting("INTERPOLATOR")
    assert (
        interpolator.source_position_name == content["misc"]["tests"]["SOURCE"]
    )
