from collections.abc import Callable
from typing import Any

from ._logging_items import (
    ILoggingItem,
    LoggingDictItem,
    LoggingFloatItem,
    LoggingIntItem,
    LoggingStrItem,
)


def create_logitems(
    value: str | float | int | dict[str, float] | None,
    title: str = None,
    default_factory: Callable[[], Any] | None = None,
) -> ILoggingItem:
    if isinstance(value, str):
        return LoggingStrItem(val=value)

    if isinstance(value, float):
        return LoggingFloatItem(val=value, title=title)

    if isinstance(value, int):
        return LoggingIntItem(val=value, title=title)

    if isinstance(value, dict):
        return LoggingDictItem(val=value, title=title)

    if value is None:
        if default_factory is None:
            return LoggingFloatItem(val=value, title=title)
        else:
            val = default_factory()
            return create_logitems(val, title)

    raise NotImplementedError(
        f"{type(value)} is not implemented as a logging item."
    )
