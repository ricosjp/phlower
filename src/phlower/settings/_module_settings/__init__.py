from phlower.settings._interface import (
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)
from phlower.settings._module_settings._accessor_setting import AccessorSetting
from phlower.settings._module_settings._concatenator_setting import (
    ConcatenatorSetting,
)
from phlower.settings._module_settings._contraction_setting import (
    ContractionSetting,
)
from phlower.settings._module_settings._deepsets_setting import DeepSetsSetting
from phlower.settings._module_settings._dirichlet_setting import (
    DirichletSetting,
)
from phlower.settings._module_settings._einsum_setting import EinsumSetting
from phlower.settings._module_settings._en_equivariant_mlp_setting import (
    EnEquivariantMLPSetting,
)
from phlower.settings._module_settings._en_equivariant_tcn_setting import (
    EnEquivariantTCNSetting,
)
from phlower.settings._module_settings._gcn_setting import GCNSetting
from phlower.settings._module_settings._identity_setting import IdentitySetting
from phlower.settings._module_settings._isogcn_setting import IsoGCNSetting
from phlower.settings._module_settings._layer_norm_setting import (
    LayerNormSetting,
)
from phlower.settings._module_settings._mlp_setting import MLPSetting
from phlower.settings._module_settings._pinv_mlp_setting import PInvMLPSetting
from phlower.settings._module_settings._pooling_setting import PoolingSetting
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
from phlower.settings._module_settings._spmm_setting import SPMMSetting
from phlower.settings._module_settings._tcn_setting import TCNSetting
from phlower.settings._module_settings._time_series_to_features_setting import (
    TimeSeriesToFeaturesSetting,
)
from phlower.settings._module_settings._transolver_attention_setting import (
    TransolverAttentionSetting,
)

_layer_settings: list[type[IPhlowerLayerParameters]] = [
    AccessorSetting,
    ConcatenatorSetting,
    DeepSetsSetting,
    DirichletSetting,
    ContractionSetting,
    EnEquivariantTCNSetting,
    EnEquivariantMLPSetting,
    EinsumSetting,
    GCNSetting,
    IdentitySetting,
    IsoGCNSetting,
    LayerNormSetting,
    MLPSetting,
    PInvMLPSetting,
    ProportionalSetting,
    RearrangeSetting,
    ReducerSetting,
    ShareSetting,
    SimilarityEquivariantMLPSetting,
    SPMMSetting,
    TCNSetting,
    TimeSeriesToFeaturesSetting,
    TransolverAttentionSetting,
    PoolingSetting,
]
