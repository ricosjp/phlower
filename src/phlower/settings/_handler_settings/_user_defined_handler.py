from typing import Any

import pydantic


class UserDefinedHandlerSetting(pydantic.BaseModel):
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
