from phlower.utils.typing import PhlowerHandlerType

from ._early_stopping import EarlyStopping

_REGISTERED: dict[str, type[PhlowerHandlerType]] = {
    "EarlyStopping": EarlyStopping
}


def create_handler(name: str, params: dict) -> PhlowerHandlerType:
    if name not in _REGISTERED:
        raise NotImplementedError(f"Handler {name} is not implemented.")
    return _REGISTERED[name](**params)
