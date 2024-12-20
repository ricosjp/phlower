from phlower.nn._core_modules._concatenator import Concatenator
from phlower.nn._core_modules._en_equivariant_mlp import EnEquivariantMLP
from phlower.nn._core_modules._gcn import GCN
from phlower.nn._core_modules._identity import Identity
from phlower.nn._core_modules._mlp import MLP
from phlower.nn._core_modules._pinv_mlp import PInvMLP
from phlower.nn._core_modules._proportional import Proportional
from phlower.nn._core_modules._share import Share
from phlower.nn._core_modules._similarity_equivariant_mlp import (
    SimilarityEquivariantMLP,
)
from phlower.nn._interface_module import IPhlowerCoreModule

_all_models: list[IPhlowerCoreModule] = [
    Concatenator,
    EnEquivariantMLP,
    GCN,
    Identity,
    MLP,
    PInvMLP,
    Proportional,
    Share,
    SimilarityEquivariantMLP,
]

_name2model = {cls.get_nn_name(): cls for cls in _all_models}


def get_module(name: str) -> IPhlowerCoreModule:
    return _name2model[name]
