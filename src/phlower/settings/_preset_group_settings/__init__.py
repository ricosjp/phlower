from phlower.settings._interface import (
    IPhlowerPresetGroupParameters,
    IReadOnlyReferenceGroupSetting,
)
from phlower.settings._preset_group_settings._identity_setting import (
    IdentityPresetGroupSetting,
)

_preset_group_layer_settings: list[type[IPhlowerPresetGroupParameters]] = [
    IdentityPresetGroupSetting,
]
