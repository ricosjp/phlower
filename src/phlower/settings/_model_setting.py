from __future__ import annotations

import itertools

import pydantic
from pipe import select, uniq
from typing_extensions import Self

from phlower._base import PhysicalDimensions, PhysicalDimensionsClass
from phlower.settings._group_settings import GroupModuleSetting


class _MemberSetting(pydantic.BaseModel):
    name: str
    n_last_dim: int | None = None

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class ModelIOSetting(pydantic.BaseModel):
    name: str
    is_time_series: bool = False
    is_voxel: bool = False
    # time_slices: slice | None = None
    physical_dimension: PhysicalDimensionsClass | None = None
    """
    physical dimension
    """

    members: list[_MemberSetting] = pydantic.Field(default_factory=list)

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _register_if_empty(cls, values: dict) -> dict:
        if (values.get("members") is None) or (len(values.get("members")) == 0):
            values["members"] = [{"name": values.get("name")}]

        return values

    @property
    def n_last_dim(self) -> int:
        return sum(v.n_last_dim for v in self.members)


class PhlowerModelSetting(pydantic.BaseModel):
    variable_dimensions: dict[str, PhysicalDimensionsClass] = pydantic.Field(
        default_factory=lambda: {}, validate_default=True
    )
    """
    dictionary which maps variable name to value
    """

    inputs: list[ModelIOSetting]
    """
    settings for input feature values
    """

    labels: list[ModelIOSetting] = pydantic.Field(
        default_factory=list, frozen=True
    )
    """
    settings for output feature value
    """

    fields: list[ModelIOSetting] = pydantic.Field(
        default_factory=list, frozen=True
    )
    """
    name of variables in simulation field which are treated as constant
     in calculation.
    For example, support matrix for input graph structure.
    """

    network: GroupModuleSetting
    """
    define structure of neural network
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )

    @pydantic.model_validator(mode="after")
    def check_duplicate_names(self) -> Self:
        unique_names = list(self.fields | select(lambda x: x.name) | uniq)

        if len(unique_names) != len(self.fields):
            raise ValueError(
                "Duplicate name is found. A varaible name "
                "in field setting must be unique."
            )

        return self

    def get_name_to_dimensions(self) -> dict[str, PhysicalDimensions | None]:
        return {
            v.name: v.physical_dimension
            for v in itertools.chain(self.inputs, self.labels, self.fields)
        }

    def resolve(self) -> None:
        """
        Resolve network relationship. Following items are checked.

        * network is a valid DAG, not cycled graph.
        * input keywards in a module is defined
          as output in the precedent module.
        * set positive integer value to the value
          which is defined as -1 in nodes.
        """
        _inputs = [{v.name: v.n_last_dim} for v in self.inputs]
        self.network.resolve(*_inputs)
