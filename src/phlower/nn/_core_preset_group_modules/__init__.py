from phlower.nn._core_preset_group_modules._identity import (
    IdentityPresetGroupModule,
)
from phlower.nn._core_preset_group_modules._interpolator import (
    InterpolatorPresetGroupModule,
)
from phlower.nn._interface_module import (
    IPhlowerCorePresetGroupModule,
)

_all_presets: list[type[IPhlowerCorePresetGroupModule]] = [
    IdentityPresetGroupModule,
    InterpolatorPresetGroupModule,
]

_name2presets = {cls.get_nn_name(): cls for cls in _all_presets}


def get_preset_group_module(name: str) -> type[IPhlowerCorePresetGroupModule]:
    return _name2presets[name]


__all__ = [m.__name__ for m in _all_presets] + ["get_preset_group_module"]
