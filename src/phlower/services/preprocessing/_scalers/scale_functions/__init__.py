from phlower.services.preprocessing._scalers import IPhlowerScaler
from phlower.utils.enums import PhlowerScalerName

from .identity_scaler import IdentityScaler
from .isoam_scaler import IsoAMScaler
from .logit_transform import LogitTransformScaler
from .max_abs_scaler import MaxAbsPoweredScaler
from .min_max_scaler import MinMaxScaler
from .standard_scaler import StandardScaler

# name to scaler class and default arguments

_alias_to_scaler: dict[str, IPhlowerScaler] = {
    PhlowerScalerName.identity.value: IdentityScaler,
    PhlowerScalerName.isoam_scale.value: IsoAMScaler,
    PhlowerScalerName.max_abs_powered.value: MaxAbsPoweredScaler,
    PhlowerScalerName.min_max.value: MinMaxScaler,
    PhlowerScalerName.standardize.value: StandardScaler,
    PhlowerScalerName.std_scale.value: StandardScaler,
    PhlowerScalerName.logit_transform.value: LogitTransformScaler,
}


def create_scaler(scaler_name: str, **kwards) -> IPhlowerScaler:
    scaler = _alias_to_scaler.get(scaler_name)
    if scaler is None:
        raise NotImplementedError(f"{scaler_name} is not defined.")

    return scaler.create(scaler_name, **kwards)
