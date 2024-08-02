from phlower.services.preprocessing._scalers import IPhlowerScaler
from phlower.utils.enums import PhlowerScalerName

from .identity_scaler import IdentityScaler
from .isoam_scaler import IsoAMScaler
from .max_abs_scaler import MaxAbsPoweredScaler
from .min_max_scaler import MinMaxScaler
from .standard_scaler import StandardScaler

# name to scaler class and default arguments

_alias_to_scaler: dict[str, IPhlowerScaler] = {
    PhlowerScalerName.IDENTITY.value: IdentityScaler,
    PhlowerScalerName.ISOAM_SCALE.value: IsoAMScaler,
    PhlowerScalerName.MAX_ABS_POWERED.value: MaxAbsPoweredScaler,
    PhlowerScalerName.MIN_MAX.value: MinMaxScaler,
    PhlowerScalerName.STANDARDIZE.value: StandardScaler,
    PhlowerScalerName.STD_SCALE.value: StandardScaler,
}


def create_scaler(scaler_name: str, **kwards) -> IPhlowerScaler:
    scaler = _alias_to_scaler.get(scaler_name)
    if scaler is None:
        raise NotImplementedError(f"{scaler_name} is not defined.")

    return scaler.create(scaler_name, **kwards)
