from typing import NamedTuple

from phlower_tensor.utils.enums import ConcatenateType


class BatchModeHolder(NamedTuple):
    inputs_batch_mode: dict[str, ConcatenateType | None]
    labels_batch_mode: dict[str, ConcatenateType | None]
    field_batch_mode: dict[str, ConcatenateType | None]
