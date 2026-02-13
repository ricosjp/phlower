from __future__ import annotations

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerPresetGroupParameters,
)


class InterpolatorPresetGroupSetting(
    IPhlowerPresetGroupParameters, pydantic.BaseModel
):
    source_position_name: str = Field(frozen=True)
    target_position_name: str = Field(frozen=True)
    source_position_from_field: bool = Field(True, frozen=True)
    target_position_from_field: bool = Field(True, frozen=True)
    variable_names: list[str] | None = Field(None, frozen=True)
    input_to_output_map: dict[str, str] | None = Field(None, frozen=True)
    source_field_name: str | None = Field(None, frozen=True)
    target_field_name: str | None = Field(None, frozen=True)
    n_head: int = Field(1, frozen=True)
    weight_name: str | None = Field(None, frozen=True)
    length_scale: float = Field(1.0e-1, frozen=True)
    normalize_row: bool = Field(True, frozen=True)
    conservative: bool = Field(False, frozen=True)
    bidirectional: bool = Field(False, frozen=True)
    k_nns: int = Field(9, frozen=True)
    d_nns: float | None = Field(None, frozen=True)
    learnable_length_scale: bool = Field(False, frozen=True)
    learnable_weight_transpose: bool = Field(False, frozen=True)
    learnable_weight_distance: bool = Field(False, frozen=True)
    recompute_distance_k_nns_if_grad_enabled: bool = Field(False, frozen=True)
    recompute_distance_d_nns_if_grad_enabled: bool = Field(False, frozen=True)
    unbatch: bool = Field(True, frozen=True)
    epsilon: float = Field(1.0e-5, frozen=True)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    def get_preset_type(cls) -> str:
        return "Interpolator"

    def confirm(self, self_module: IModuleSetting) -> None:
        return
