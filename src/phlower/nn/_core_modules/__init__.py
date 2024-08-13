from phlower.nn._core_modules._concatenator import Concatenator
from phlower.nn._core_modules._gcn import GCN
from phlower.nn._core_modules._mlp import MLP
from phlower.nn._core_modules._share import Share
from phlower.nn._interface_module import IPhlowerCoreModule

_all_models: list[IPhlowerCoreModule] = [GCN, MLP, Concatenator, Share]

_name2model = {cls.get_nn_name(): cls for cls in _all_models}


def get_module(name: str) -> IPhlowerCoreModule:
    return _name2model[name]
