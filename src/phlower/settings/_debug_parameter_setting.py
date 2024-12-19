import pydantic
from pydantic import Field


class PhlowerModuleDebugParameters(pydantic.BaseModel):
    output_tensor_shape: list[int] | None = Field(None, frozen=True)
