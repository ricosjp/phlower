from __future__ import annotations

import torch
from typing_extensions import Self

from phlower._fields import ISimulationField
from phlower.collections.tensors import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.nn._interface_module import (
    IPhlowerCorePresetGroupModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._preset_group_settings import IdentityPresetGroupSetting


class IdentityPresetGroupModule(torch.nn.Module, IPhlowerCorePresetGroupModule):
    @classmethod
    def from_setting(cls, setting: IdentityPresetGroupSetting) -> Self:
        return IdentityPresetGroupModule(**setting.model_dump())

    @classmethod
    def get_nn_name(cls) -> str:
        return "Identity"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        input_to_output_map: dict[str, str],
    ) -> None:
        super().__init__()
        self._input_to_output = input_to_output_map

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None: ...

    def get_reference_name(self) -> str | None:
        return None

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None = None,
        **kwards,
    ) -> IPhlowerTensorCollections:
        ans = phlower_tensor_collection(
            {
                output_key: data[input_key]
                for input_key, output_key in self._input_to_output.items()
            }
        )

        return ans
