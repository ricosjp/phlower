import pathlib
from typing import Literal

import pydantic


class CalculationState(pydantic.BaseModel):
    mode: Literal["training", "validation", "prediction"]
    output_directory: pathlib.Path | None = None
    current_epoch: int | None = None
    current_batch_iteration: int | None = None

    model_config = pydantic.ConfigDict(
        extra="forbid",
        frozen=True,
    )
