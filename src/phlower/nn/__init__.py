from phlower.nn._interface_layer import IPhlowerLayerParameters
from phlower.nn._modules._concatenator import Concatenator, ConcatenatorSetting
from phlower.nn._modules._gcn import GCN, GCNSetting
from phlower.nn._modules._mlp import MLP, MLPSetting

_NAME_TO_SETTINGS: dict[str, IPhlowerLayerParameters] = {
    GCN.get_nn_name(): GCNSetting,
    MLP.get_nn_name(): MLPSetting,
    Concatenator.get_nn_name(): ConcatenatorSetting,
}


def check_exist_module(name: str) -> bool:
    return name in _NAME_TO_SETTINGS


def gather_input_dims(name: str, *input_dims: int):
    setting = _NAME_TO_SETTINGS[name]
    return setting.gather_input_dims(*input_dims)


def parse_parameter_setting(name: str, **kwards) -> IPhlowerLayerParameters:
    return _NAME_TO_SETTINGS[name](**kwards)
