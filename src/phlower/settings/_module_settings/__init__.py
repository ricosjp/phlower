from phlower.settings._module_settings._concatenator_setting import (
    ConcatenatorSetting,
)
from phlower.settings._module_settings._gcn_setting import GCNSetting
from phlower.settings._module_settings._mlp_setting import MLPSetting
from phlower.settings._module_settings._interface import IPhlowerLayerParameters

_name_to_setting: dict[str, IPhlowerLayerParameters] = {
    "GCN": GCNSetting,
    "MLP": MLPSetting,
    "Concatenator": ConcatenatorSetting,
}


def check_exist_module(name: str) -> bool:
    return name in _name_to_setting


def gather_input_dims(name: str, *input_dims: int):
    setting = _name_to_setting[name]
    return setting.gather_input_dims(*input_dims)

