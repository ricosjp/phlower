from phlower.nn._core_preset_group_modules._identity import (
    IdentityPresetGroupModule,
)
from phlower.nn._interface_module import (
    IPhlowerCorePresetGroupModule,
)

_all_models: list[type[IPhlowerCorePresetGroupModule]] = [
    IdentityPresetGroupModule,
]

_name2model = {cls.get_nn_name(): cls for cls in _all_models}


def get_preset_group_module(name: str) -> type[IPhlowerCorePresetGroupModule]:
    return _name2model[name]


__all__ = list(_name2model.keys()) + ["get_preset_group_module"]
