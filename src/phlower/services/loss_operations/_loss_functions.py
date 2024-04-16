
from phlower.utils.typing import LossFunctionType

import torch


_DEFAULT_LOSS_FUNCTIONS: dict[str, LossFunctionType] = {
    "mse": torch.nn.functional.mse_loss
}


def get_loss_function(name: str) -> LossFunctionType | None:
    return _DEFAULT_LOSS_FUNCTIONS.get(name)
