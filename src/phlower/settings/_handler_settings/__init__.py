from enum import Enum
from typing import Annotated

from pydantic import Discriminator, Tag

from phlower.settings._handler_settings._early_stopping import (
    EarlyStoppingSetting,
)
from phlower.settings._handler_settings._user_defined_handler import (
    UserDefinedHandlerSetting,
)


class _DiscriminatorHandlerTag(str, Enum):
    EarlyStopping = "EarlyStopping"
    UserCustom = "UserCustom"


def _custom_handler_discriminator(value: object) -> str:
    if isinstance(value, EarlyStoppingSetting):
        return _DiscriminatorHandlerTag.EarlyStopping.value
    if isinstance(value, UserDefinedHandlerSetting):
        return _DiscriminatorHandlerTag.UserCustom.value

    handler_type = value.get("handler", None)

    if not isinstance(handler_type, str):
        raise ValueError(
            f"Invalid type value is set as a handler. Input: {handler_type}"
        )

    if handler_type == _DiscriminatorHandlerTag.EarlyStopping:
        return _DiscriminatorHandlerTag.EarlyStopping.value
    return _DiscriminatorHandlerTag.UserCustom.value


HandlerSettingType = Annotated[
    Annotated[
        EarlyStoppingSetting,
        Tag(_DiscriminatorHandlerTag.EarlyStopping.value),
    ]
    | Annotated[
        UserDefinedHandlerSetting,
        Tag(_DiscriminatorHandlerTag.UserCustom.value),
    ],
    Discriminator(
        _custom_handler_discriminator,
        custom_error_type="invalid_union_member",
        custom_error_message="Invalid union member",
        custom_error_context={"discriminator": "handler_checkk"},
    ),
]
