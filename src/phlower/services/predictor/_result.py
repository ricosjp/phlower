from typing import NamedTuple

import numpy as np
from phlower_tensor.collections import IPhlowerTensorCollections


class PhlowerInverseScaledPredictionResult(NamedTuple):
    prediction_data: dict[str, np.ndarray]
    input_data: dict[str, np.ndarray]
    answer_data: dict[str, np.ndarray] | None = None


class PhlowerPredictionResult(NamedTuple):
    prediction_data: IPhlowerTensorCollections
    input_data: IPhlowerTensorCollections | None = None
    answer_data: IPhlowerTensorCollections | None = None
