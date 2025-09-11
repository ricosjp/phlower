from __future__ import annotations

import pydantic
from pydantic import Field

from phlower.settings._interface import (
    IModuleSetting,
    IPhlowerLayerParameters,
    IReadOnlyReferenceGroupSetting,
)
from phlower.utils.enums import PoolingType


class PoolingSetting(IPhlowerLayerParameters, pydantic.BaseModel):
    unbatch_key: str | None = Field(None, frozen=True)
    nodes: list[int] | None = Field(None)
    pool_operator_name: PoolingType | str = Field("max", frozen=True)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    def get_nn_type(cls) -> str:
        return "Pooling"

    @pydantic.field_serializer("pool_operator_name")
    def serialize_pool_operator(self, value: PoolingType | str) -> str:
        if isinstance(value, PoolingType):
            return value.name

        return value

    @pydantic.field_validator("nodes", mode="before")
    @classmethod
    def check_n_nodes(cls, vals: list[int]) -> list[int]:
        if vals is None:
            return vals

        if len(vals) != 2:
            raise ValueError(
                "size of nodes must be 2 in PoolingSetting." f" input: {vals}"
            )
        return vals

    @pydantic.field_validator("pool_operator_name", mode="before")
    def check_exist_pooling_operator(cls, value: str) -> PoolingType:
        """Check if the pooling operator is valid or not.

        Raises:
            ValueError: If the pooling operator is not valid.
        """

        if isinstance(value, PoolingType):
            return value

        if value not in PoolingType.__members__:
            raise ValueError(f"Pooling operator {value} " "is not implemented.")
        return PoolingType[value]

    def gather_input_dims(self, *input_dims: int) -> int:
        """Gather feature dimensions of input tensors

        Args:
            *input_dims (list[int]): feature dimensions of input tensors

        Returns:
            int: resolved dimension of input tensors
        """
        if len(input_dims) != 1:
            raise ValueError("only one input is allowed in Pooling.")
        return input_dims[0]

    def get_default_nodes(self, *input_dims: int) -> list[str]:
        """Return desired nodes when nodes is None
        Returns:
            list[str]: default nodes
        """
        n_dim = self.gather_input_dims(*input_dims)
        return [n_dim, n_dim]

    def get_n_nodes(self) -> list[int] | None:
        """Return feature dimensions inside PhlowerLayer.

        Returns:
            list[int] | None: feature dimensions
        """
        return self.nodes

    def overwrite_nodes(self, nodes: list[int]) -> None:
        """overwrite feature dimensions by using resolved information

        Args:
            nodes (list[int]):
                feature dimensions which is resolved
                 by using status of precedent and succedent phlower modules.
        """
        self.nodes = nodes

    @property
    def need_reference(self) -> bool:
        """Whether the reference to other phlower modules is needed or not.

        Returns:
            bool: True if reference to other phlower modules is necessary.
        """
        return False

    def get_reference(self, parent: IReadOnlyReferenceGroupSetting):
        """Get reference information from parent group setting

        Args:
            parent (IReadOnlyReferenceGroupSetting):
                Reference to group setting of its parent
        """
        return

    def confirm(self, self_module: IModuleSetting) -> None:
        """Chenck and confirm parameters. This functions is
         called after all values of parameters are established.
         Write scripts to check its state at last.

        Args:
            input_keys (list[str]): keys to input this parameters
            **kwards: information of parent module
        """
        ...
