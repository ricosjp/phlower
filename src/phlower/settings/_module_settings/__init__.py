from phlower.settings._interface import (
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)
from phlower.settings._module_settings._concatenator_setting import (
    ConcatenatorSetting,
)
from phlower.settings._module_settings._gcn_setting import GCNSetting
from phlower.settings._module_settings._mlp_setting import MLPSetting
from phlower.settings._module_settings._share_setting import ShareSetting

_name_to_setting: dict[str, IPhlowerLayerParameters] = {
    "GCN": GCNSetting,
    "MLP": MLPSetting,
    "Concatenator": ConcatenatorSetting,
    "Share": ShareSetting,
}


def check_exist_module(name: str) -> bool:
    return name in _name_to_setting
