from __future__ import annotations

from typing import Annotated, Any

import pydantic
from pydantic import Field

from phlower.utils import (
    OptimizerSelector,
    get_logger,
)

_logger = get_logger(__name__)


_ParameterValue = Annotated[
    int | float | bool | str | Any, Field(union_mode="left_to_right")
]


class OptimizerSetting(pydantic.BaseModel):
    optimizer: str = "Adam"
    """
    Optimizer Class name defined in torch.optim. Default to Adam.
    Ex. Adam, RMSprop, SGD
    """

    parameters: dict[str, _ParameterValue] = Field(default_factory=dict)
    """
    Parameters to pass when optimizer class is initialized.
    Allowed parameters depend on the optimizer you choose.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    def get_lr(self) -> float | None:
        return self.parameters.get("lr")

    @pydantic.field_validator("optimizer")
    @classmethod
    def check_exist_scheduler(cls, name: str) -> str:
        if not OptimizerSelector.exist(name):
            raise ValueError(
                f"{name} is not defined as an optimizer in phlower. "
                "If you defined user defined optimizer, "
                "please use `register` function in "
                "`phlower.utils.OptimizerSelector`."
            )
        return name
