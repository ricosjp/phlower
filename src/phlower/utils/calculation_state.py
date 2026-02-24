import dataclasses as dc
import pathlib
from typing import Literal


@dc.dataclass(frozen=True, slots=True)
class CalculationState:
    mode: Literal["training", "validation"]
    output_directory: pathlib.Path | None = None
    current_epoch: int | None = None
    current_batch_iteration: int | None = None
