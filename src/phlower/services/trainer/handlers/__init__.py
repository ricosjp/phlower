from ._early_stopping import EarlyStopping
from ._interface import IHandlerCall

_REGISTERED: dict[str, type[IHandlerCall]] = {
    EarlyStopping.name(): EarlyStopping
}


def register_user_handler(
    name: str, handler: IHandlerCall, force_update: bool = False
) -> None:
    if (name in _REGISTERED) and (not force_update):
        raise ValueError(f"{name} has already existed.")

    _REGISTERED[name] = handler


def create_handler(name: str, params: dict) -> IHandlerCall:
    if name not in _REGISTERED:
        raise NotImplementedError(f"Handler {name} is not implemented.")
    return _REGISTERED[name](**params)
