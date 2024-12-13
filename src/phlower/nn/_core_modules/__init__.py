from phlower.nn._core_modules._accessor import Accessor
from phlower.nn._core_modules._concatenator import Concatenator
from phlower.nn._core_modules._contraction import Contraction
from phlower.nn._core_modules._dirichlet import Dirichlet
from phlower.nn._core_modules._einsum import Einsum
from phlower.nn._core_modules._en_equivariant_mlp import EnEquivariantMLP
from phlower.nn._core_modules._gcn import GCN
from phlower.nn._core_modules._identity import Identity
from phlower.nn._core_modules._iso_gcn import IsoGCN
from phlower.nn._core_modules._mlp import MLP
from phlower.nn._core_modules._pinv_mlp import PInvMLP
from phlower.nn._core_modules._proportional import Proportional
from phlower.nn._core_modules._rearrange import Rearrange
from phlower.nn._core_modules._reducer import Reducer
from phlower.nn._core_modules._share import Share
from phlower.nn._core_modules._similarity_equivariant_mlp import (
    SimilarityEquivariantMLP,
)
from phlower.nn._core_modules._spmm import SPMM
from phlower.nn._core_modules._time_series_to_features import (
    TimeSeriesToFeatures,
)
from phlower.nn._core_modules._utils import ActivationSelector
from phlower.nn._interface_module import IPhlowerCoreModule

_all_models: list[type[IPhlowerCoreModule]] = [
    Accessor,
    Concatenator,
    Dirichlet,
    Contraction,
    EnEquivariantMLP,
    Einsum,
    GCN,
    Identity,
    IsoGCN,
    MLP,
    PInvMLP,
    Proportional,
    Rearrange,
    Reducer,
    Share,
    SimilarityEquivariantMLP,
    TimeSeriesToFeatures,
    SPMM,
]

_name2model = {cls.get_nn_name(): cls for cls in _all_models}


def get_module(name: str) -> type[IPhlowerCoreModule]:
    return _name2model[name]
