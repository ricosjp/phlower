import pathlib

import pytest

from phlower.nn import PhlowerGroupModule
from phlower.settings import PhlowerSetting

_SAMPLE_SETTING_DIR = pathlib.Path("tests/samples/settings")


@pytest.mark.parametrize("yaml_file", ["with_share_nn.yml"])
def test__resolve_modules_from_setting(yaml_file):
    setting_file = _SAMPLE_SETTING_DIR / yaml_file

    setting = PhlowerSetting.read_yaml(setting_file)

    setting.model.network.resolve(is_first=True)
    _ = PhlowerGroupModule.from_setting(setting.model.network)
