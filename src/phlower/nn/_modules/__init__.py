import abc
from typing import Any

from phlower.nn._interface_layer import IPhlowerLayerParameters
from phlower.nn._modules._gcn import GCN, GCNResolvedSetting

_NAME_TO_SETTINGS: dict[str, Any] = {GCN.__name__: GCNResolvedSetting}


def parse_parameter_setting(name: str, **kwards) -> IPhlowerLayerParameters:
    return _NAME_TO_SETTINGS[name](**kwards)
