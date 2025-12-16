from typing import Any

import numpy as np

from phlower.utils.enums import PhlowerScalerName
from phlower.utils.typing import DenseArrayType


def get_registered_scaler_names() -> list[str]:
    return [v.value for v in PhlowerScalerName]


def convert_to_dumped(v: Any) -> Any:  # noqa: ANN401
    if isinstance(v, DenseArrayType):
        return v.tolist()

    if isinstance(v, np.generic):
        return v.item()

    if isinstance(v, str | float | int | bool):
        return v

    if isinstance(v, tuple):
        return list(v)

    if isinstance(v, list | dict):
        return v

    raise NotImplementedError(f"Conversion of {type(v)} is not implemented.")
