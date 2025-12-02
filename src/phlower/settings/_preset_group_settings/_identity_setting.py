from __future__ import annotations

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerPresetGroupParameters,
)


class IdentityPresetGroupSetting(
    IPhlowerPresetGroupParameters, pydantic.BaseModel
):
    input_to_output_map: dict[str, str] = Field(..., frozen=True)
    """
    A map from input keys to output keys.
    For example, {"input1": "output1", "input2": "output2
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    def get_preset_type(cls) -> str:
        return "Identity"

    def confirm(self, self_module: IModuleSetting) -> None:
        for key in self_module.get_input_keys():
            if key not in self.input_to_output_map:
                raise ValueError(
                    f"Mapping for input key '{key}' is not found "
                    f"in {self_module.get_name()}"
                )

        output_info = self_module.get_output_info()
        _values = self.input_to_output_map.values()
        for key in output_info.keys():
            if key not in _values:
                raise ValueError(
                    f"Mapping for output key '{key}' is not found "
                    f"in {self_module.get_name()}"
                )
        return
