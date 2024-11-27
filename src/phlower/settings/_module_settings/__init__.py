from phlower.settings._interface import (
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)
from phlower.settings._module_settings._accessor_setting import AccessorSetting
from phlower.settings._module_settings._concatenator_setting import (
    ConcatenatorSetting,
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
from phlower.settings._module_settings._reducer_setting import ReducerSetting
from phlower.settings._module_settings._share_setting import ShareSetting
from phlower.settings._module_settings._similarity_equivariant_mlp_setting import (  # noqa: E501
    SimilarityEquivariantMLPSetting,
)

_name_to_setting: dict[str, IPhlowerLayerParameters] = {
    "Accessor": AccessorSetting,
    "Concatenator": ConcatenatorSetting,
    "EnEquivariantMLP": EnEquivariantMLPSetting,
    "GCN": GCNSetting,
    "Identity": IdentitySetting,
    "MLP": MLPSetting,
    "PInvMLP": PInvMLPSetting,
    "Proportional": ProportionalSetting,
    "Reducer": ReducerSetting,
    "Share": ShareSetting,
    "SimilarityEquivariantMLP": SimilarityEquivariantMLPSetting,
    "IsoGCN": IsoGCNSetting,
}


def check_exist_module(name: str) -> bool:
    return name in _name_to_setting
