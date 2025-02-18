import torch

from phlower.utils.typing import LossFunctionType


class PhlowerLossFunctionsFactory:
    _REGISTERED: dict[str, LossFunctionType] = {
        "mse": torch.nn.functional.mse_loss
    }

    @classmethod
    def register(
        cls, name: str, loss_function: LossFunctionType, overwrite: bool = True
    ) -> LossFunctionType:
        if (name not in cls._REGISTERED) or overwrite:
            cls._REGISTERED[name] = loss_function
            return

        raise ValueError(
            f"Loss function named {name} has already existed."
            " If you want to overwrite it, set overwrite=True"
        )

    @classmethod
    def unregister(cls, name: str):
        if name not in cls._REGISTERED:
            raise KeyError(f"{name} does not exist.")

        cls._REGISTERED.pop(name)

    @classmethod
    def get(cls, name: str) -> LossFunctionType:
        if name not in cls._REGISTERED:
            raise KeyError(f"Loss function: {name} is not registered.")
        return cls._REGISTERED[name]
