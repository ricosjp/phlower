from __future__ import annotations

import itertools
from functools import cached_property

import pydantic
from pipe import select, uniq
from typing_extensions import Self

from phlower._base import PhysicalDimensions, PhysicalDimensionsClass
from phlower.settings._group_setting import GroupModuleSetting


class _MemberSetting(pydantic.BaseModel):
    name: str
    """
    Name of this feature array
    """

    n_last_dim: int | None = None
    """
    Number of feature dimensions in the last axis.
    If None is set, the dimension is not fixed.
    """

    index_first_dim: int | None = None
    """
    If set, this feature array is extracted by indexing the first axis
    with this value. Otherwise, the whole array is used.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class ModelIOSetting(pydantic.BaseModel):
    name: str
    """
    Name of this feature array
    """

    is_time_series: bool = False
    """
    Flag for time series array. Default to be False.
    """

    is_voxel: bool = False
    """
    Flag for voxel shaped array. Default to be False.
    """

    physical_dimension: PhysicalDimensionsClass | None = None
    """
    physical dimension
    """

    members: list[_MemberSetting] = pydantic.Field(default_factory=list)
    """
    List of arrays which construct this feature array
    """

    time_slice: list[int | None] | None = pydantic.Field(None, frozen=True)
    """
    slice object for time slicing
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _register_if_empty(cls, values: dict) -> dict:
        if (values.get("members") is None) or (len(values.get("members")) == 0):
            values["members"] = [{"name": values.get("name")}]

        return values

    @pydantic.field_validator("time_slice")
    def check_valid_time_slice(
        cls, value: list[int | None] | None
    ) -> slice[int | None, int | None, int | None] | None:
        if value is None:
            return value

        try:
            _ = slice(*value)

            # HACK: In Python3.10 and pydantic 2.8.2,
            #  typing definition 'slice[int, int, int]'
            #  is not allowed. Thus, this roundabout way is applied.
            return value
        except Exception as ex:
            raise ValueError(
                f"{value} cannot be converted to valid slice object."
            ) from ex

    @pydantic.model_validator(mode="after")
    def check_time_series(self) -> Self:
        if (self.time_slice is not None) and (not self.is_time_series):
            raise ValueError(
                "When using time_slice in inputs, set is_time_series as True"
            )

        return self

    @cached_property
    def _contain_none(self) -> bool:
        dims = [m.n_last_dim for m in self.members]
        return None in dims

    @property
    def time_slice_object(self) -> tuple[int, int, int]:
        # NOTE: check comment in 'check_valid_time_slice'
        return slice(*self.time_slice)

    def has_member(self, name: str) -> bool:
        """
        Check if this setting has member with the specified name.
        """
        return any(m.name == name for m in self.members)

    def get_n_last_dim(self) -> int | None:
        if self._contain_none:
            # "Feature dimensions cannot be calculated"
            # " because n_last_dim set to be None"
            # " among these members."
            return None
        return sum(v.n_last_dim for v in self.members)


class PhlowerModelSetting(pydantic.BaseModel):
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
        _inputs = [{v.name: v.get_n_last_dim()} for v in self.inputs]
        self.network.resolve(*_inputs)
