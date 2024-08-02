from typing import Annotated

import pydantic
from pydantic import (
    PlainSerializer,
    PlainValidator,
    SerializationInfo,
    ValidationInfo,
)

from phlower.settings._module_settings import (
    IPhlowerLayerParameters,
    _name_to_setting,
)


def _validate(vals: dict, info: ValidationInfo) -> IPhlowerLayerParameters:
    name = info.data["nn_type"]
    setting_cls = _name_to_setting[name]
    return setting_cls(**vals)


def _serialize(v: pydantic.BaseModel, info: SerializationInfo):
    return v.model_dump()


PhlowerModuleParameters = Annotated[
    IPhlowerLayerParameters,
    PlainValidator(_validate),
    PlainSerializer(_serialize),
]
