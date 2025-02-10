from ._early_stopping import EarlyStopping
from ._interface import IHandlerCall

_REGISTERED: dict[str, type[IHandlerCall]] = {
    EarlyStopping.name(): EarlyStopping
}


def create_handler(name: str, params: dict) -> IHandlerCall:
    if name not in _REGISTERED:
        raise NotImplementedError(f"Handler {name} is not implemented.")
    return _REGISTERED[name](**params)
