from phlower.services.preprocessing import IPhlowerScaler

from .identity_scaler import IdentityScaler
from .isoam_scaler import IsoAMScaler
from .max_abs_scaler import MaxAbsScaler
from .min_max_scaler import MinMaxScaler
from .sparse_standard_scaler import SparseStandardScaler
from .standard_scaler import StandardScaler

# name to scaler class and default arguments

_registered_scaler: list[IPhlowerScaler] = [
    IdentityScaler,
    IsoAMScaler,
    MaxAbsScaler,
    MinMaxScaler,
    SparseStandardScaler,
    StandardScaler
]


def get_registered_scaler_names() -> list[str]:
    names = [v.get_registered_names() for v in _registered_scaler]
    return sum(names, [])


def create_scaler(scaler_name: str, **kwards) -> IPhlowerScaler:
    for cls in _registered_scaler:
        if scaler_name in cls.get_registered_names():
            return cls.create(scaler_name, **kwards)

    raise NotImplementedError(
        f"{scaler_name} is not defined."
    )
