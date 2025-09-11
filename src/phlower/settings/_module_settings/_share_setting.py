from __future__ import annotations

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)
from phlower.utils.exceptions import NotFoundReferenceModuleError


class ShareSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    reference_name: str = Field(..., frozen=True)
    reference: IPhlowerLayerParameters | None = Field(None, exclude=True)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(
        extra="forbid", arbitrary_types_allowed=True
    )

    @classmethod
    def get_nn_type(cls) -> str:
        return "Share"

    def confirm(self, self_module: IModuleSetting) -> None:
        return

    def gather_input_dims(self, *input_dims: int) -> int:
        return self.reference.gather_input_dims(*input_dims)

    def get_default_nodes(self, *input_dims: int) -> list[int]:
        return self.reference.get_n_nodes()

    def get_n_nodes(self) -> list[int] | None:
        return self.reference.get_n_nodes()

    def check_exist_reference(self) -> None:
        if self.reference is not None:
            return

        raise ValueError(
            f"Reference setting {self.reference_name} in Share Module is None."
            "Please check that `get_reference` method has been called."
        )

    def overwrite_nodes(self, nodes: list[int]) -> None: ...

    @property
    def need_reference(self) -> bool:
        return True

    def get_reference(self, parent: IReadOnlyReferenceGroupSetting):
        try:
            self.reference = parent.search_module_setting(self.reference_name)
        except KeyError as ex:
            raise NotFoundReferenceModuleError(
                f"Reference module {self.reference_name} is not found "
                "in the same group."
            ) from ex
