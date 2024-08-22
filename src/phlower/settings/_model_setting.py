from __future__ import annotations

import pydantic
from pipe import uniq
from typing_extensions import Self

from phlower._base import PhysicalDimensionsClass
from phlower.settings._group_settings import GroupModuleSetting


class PhlowerFieldSetting(pydantic.BaseModel):
    field_names: list[str] = pydantic.Field(default_factory=list, frozen=True)
    """
    name of variables in simulation field which are treated as constant
     in calculation.
    For example,

        * support matrix for input graph structure
        * boundary conditions

    """

    @pydantic.model_validator(mode="after")
    def check_duplicate_names(self) -> Self:
        unique_names = list(self.field_names | uniq)

        if len(unique_names) != len(self.field_names):
            raise ValueError(
                "Duplicate name is found. A varaible name "
                "in field setting must be unique."
            )

        return self


class PhlowerModelSetting(pydantic.BaseModel):
    variable_dimensions: dict[str, PhysicalDimensionsClass] = pydantic.Field(
        default_factory=lambda: {}, validate_default=True
    )
    """
    dictionary which maps variable name to value
    """

    network: GroupModuleSetting
    """
    define structure of neural network
    """

    fields: PhlowerFieldSetting = pydantic.Field(
        default_factory=PhlowerFieldSetting
    )
    """
    settings for fields dependent on your mesh or graph
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )

    def resolve(self) -> None:
        """
        Resolve network relationship. Following items are checked.

        * network is a valid DAG, not cycled graph.
        * input keywards in a module is defined
          as output in the precedent module.
        * set positive integer value to the value
          which is defined as -1 in nodes.
        """
        self.network.resolve(is_first=True)
