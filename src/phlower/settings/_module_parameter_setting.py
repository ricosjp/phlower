from typing import Annotated, Any, TypeAlias

import pydantic
from pydantic import (
    PlainSerializer,
    PlainValidator,
    SerializationInfo,
    ValidationInfo,
)

from phlower.settings._module_settings import (
    IPhlowerLayerParameters,
    _layer_settings,
)

_name_to_setting = {s.get_nn_type(): s for s in _layer_settings}

EmptyDict: TypeAlias = dict


def _validate(
    vals: dict, info: ValidationInfo
) -> IPhlowerLayerParameters | EmptyDict:
    if info.data["nn_parameters_same_as"] is not None:
        if len(vals) != 0:
            raise ValueError(
                "nn_parameters must be empty when nn_parameters_same_as is set."
            )
        return {}

    if "nn_type" not in info.data:
        raise ValueError(f"nn_type is not defined in {info.data['name']}")
    name = info.data["nn_type"]

    if name not in _name_to_setting:
        raise ValueError(
            f"nn_type={name} is not implemented. ({info.data['name']})"
        )
    setting_cls = _name_to_setting[name]
    return setting_cls(**vals)


def _serialize(
    v: pydantic.BaseModel, info: SerializationInfo
) -> dict[str, Any]:
    return v.model_dump()


PhlowerModuleParameters = Annotated[
    IPhlowerLayerParameters | EmptyDict,
    PlainValidator(_validate),
    PlainSerializer(_serialize),
]
