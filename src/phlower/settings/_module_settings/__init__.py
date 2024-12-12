from phlower.settings._interface import (
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)
from phlower.settings._module_settings._accessor_setting import AccessorSetting
from phlower.settings._module_settings._concatenator_setting import (
    ConcatenatorSetting,
)
from phlower.settings._module_settings._dirichlet_setting import (
    DirichletSetting,
)
from phlower.settings._module_settings._en_equivariant_mlp_setting import (
    EnEquivariantMLPSetting,
)
from phlower.settings._module_settings._gcn_setting import GCNSetting
from phlower.settings._module_settings._identity_setting import IdentitySetting
from phlower.settings._module_settings._isogcn_setting import IsoGCNSetting
from phlower.settings._module_settings._mlp_setting import MLPSetting
from phlower.settings._module_settings._pinv_mlp_setting import PInvMLPSetting
from phlower.settings._module_settings._proportional_setting import (
    ProportionalSetting,
)
from phlower.settings._module_settings._rearrange_setting import (
    RearrangeSetting,
)
from phlower.settings._module_settings._reducer_setting import ReducerSetting
from phlower.settings._module_settings._share_setting import ShareSetting
from phlower.settings._module_settings._similarity_equivariant_mlp_setting import (  # noqa: E501
    SimilarityEquivariantMLPSetting,
)
from phlower.settings._module_settings._time_series_to_features_setting import (
    TimeSeriesToFeaturesSetting,
)

_name_to_setting: dict[str, IPhlowerLayerParameters] = {
    "Accessor": AccessorSetting,
    "Concatenator": ConcatenatorSetting,
    "Dirichlet": DirichletSetting,
    "EnEquivariantMLP": EnEquivariantMLPSetting,
    "GCN": GCNSetting,
    "Identity": IdentitySetting,
    "IsoGCN": IsoGCNSetting,
    "MLP": MLPSetting,
    "PInvMLP": PInvMLPSetting,
    "Proportional": ProportionalSetting,
    "Rearrange": RearrangeSetting,
    "Reducer": ReducerSetting,
    "Share": ShareSetting,
    "SimilarityEquivariantMLP": SimilarityEquivariantMLPSetting,
    "TimeSeriesToFeatures": TimeSeriesToFeaturesSetting,
}


def check_exist_module(name: str) -> bool:
    return name in _name_to_setting
