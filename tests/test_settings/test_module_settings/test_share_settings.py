import pathlib

import pytest
import yaml

from phlower.settings import PhlowerModelSetting, PhlowerSetting

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/share_settings"


def test__can_resolve():
    setting = PhlowerSetting.read_yaml(_TEST_DATA_DIR / "with_share_nn.yml")
    setting.model.network.resolve(is_first=True)


@pytest.mark.parametrize(
    "yaml_file", ["check_gcn_share_nodes.yml", "check_mlp_share_nodes.yml"]
)
def test__nodes_after_resolve(yaml_file):
    with open(_TEST_DATA_DIR / yaml_file) as fr:
        content = yaml.load(fr, Loader=yaml.SafeLoader)

    setting = PhlowerModelSetting(**content["model"])
    setting.network.resolve(is_first=True)

    assert len(content["misc"]["tests"].items()) > 0

    for key, value in content["misc"]["tests"].items():
        target = setting.network.search_module_setting(key)
        assert target.get_n_nodes() == value
