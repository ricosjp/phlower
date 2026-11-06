from __future__ import annotations

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)


class NodalExpandSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    """Expand the other input to the node count of the named reference."""

    reference_key: str = Field(frozen=True)
    """
    Name of the reference tensor in input_keys.
    """
    nodes: list[int] | None = Field(None)
    unbatch_key: str | None = Field(None, frozen=True)
    """
    Batch information key used to expand each sample separately.
    """

    model_config = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    def get_nn_type(cls) -> str:
        return "NodalExpand"

    def gather_input_dims(self, input_dims: dict[str, int]) -> int:
        if len(input_dims) != 2:
            raise ValueError("Exactly two inputs are required in NodalExpand.")

        if self.reference_key not in input_dims:
            raise ValueError(
                f"reference_key {self.reference_key!r} "
                "is not found in input_keys."
            )
        return next(
            value
            for key, value in input_dims.items()
            if key != self.reference_key
        )

    def get_default_nodes(self, input_dims: dict[str, int]) -> list[int]:
        n_dim = self.gather_input_dims(input_dims)
        return [n_dim, n_dim]

    @pydantic.field_validator("nodes", mode="before")
    @classmethod
    def check_n_nodes(cls, vals: list[int] | None) -> list[int] | None:
        if vals is not None and len(vals) != 2:
            raise ValueError(
                f"Size of nodes must be 2 in NodalExpandSetting. input: {vals}"
            )
        return vals

    def get_n_nodes(self) -> list[int] | None:
        return self.nodes

    def overwrite_nodes(self, nodes: list[int]) -> None:
        self.nodes = nodes

    @property
    def need_reference(self) -> bool:
        return False

    def get_reference(self, parent: IReadOnlyReferenceGroupSetting) -> None:
        return

    def confirm(self, self_module: IModuleSetting) -> None:
        self.gather_input_dims(dict.fromkeys(self_module.get_input_keys(), 0))
