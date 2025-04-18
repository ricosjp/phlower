from typing import Any

import pydantic
from typing_extensions import Self


class EarlyStoppingSetting(pydantic.BaseModel):
    handler: str
    """
    handler Class name defined in phlower.services.trainer.handlers.
    """

    parameters: dict[str, Any] = pydantic.Field(default_factory=dict)
    """
    Parameters to pass when handler class is initialized.
    Allowed parameters depend on the handler you choose.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    def get_patience(self) -> int:
        return self.parameters["patience"]

    @pydantic.model_validator(mode="after")
    def check_args(self) -> Self:
        if "patience" not in self.parameters:
            raise ValueError(
                "patience keyword is necessary to define EarlyStopping."
            )

        return self
