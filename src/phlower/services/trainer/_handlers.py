from __future__ import annotations

from collections.abc import Mapping

from phlower.services.trainer._handler_functions import EarlyStopping
from phlower.settings import PhlowerTrainerSetting
from phlower.utils.enums import PhlowerHandlerRegisteredKey
from phlower.utils.typing import AfterEvaluationOutput, PhlowerHandlerType


class PhlowerHandlersFactory:
    _REGISTERED: dict[str, type[PhlowerHandlerType]] = {
        "EarlyStopping": EarlyStopping
    }

    @classmethod
    def register(
        cls,
        name: str,
        handler_type: type[PhlowerHandlerType],
        overwrite: bool = False,
    ):
        if (name not in cls._REGISTERED) or overwrite:
            cls._REGISTERED[name] = handler_type
            return

        raise ValueError(
            f"Handler named {name} has already existed."
            " If you want to overwrite it, set overwrite=True"
        )

    @classmethod
    def unregister(cls, name: str):
        if name not in cls._REGISTERED:
            raise KeyError(f"{name} does not exist.")

        cls._REGISTERED.pop(name)

    @classmethod
    def create(cls, name: str, params: dict) -> PhlowerHandlerType:
        if name not in cls._REGISTERED:
            raise NotImplementedError(f"Handler {name} is not registered.")
        return cls._REGISTERED[name](**params)


class PhlowerHandlersRunner:
    @classmethod
    def from_setting(
        cls, setting: PhlowerTrainerSetting
    ) -> PhlowerHandlersRunner:
        return PhlowerHandlersRunner(
            handlers_params={
                v.handler: v.parameters for v in setting.handler_settings
            }
        )

    def __init__(
        self,
        handlers_params: dict[str, dict],
    ):
        self._terminate = False
        self._handlers = {
            name: PhlowerHandlersFactory.create(name, params)
            for name, params in handlers_params.items()
        }

    @property
    def n_handlers(self) -> int:
        return len(self._handlers)

    @classmethod
    def name(cls) -> str:
        return "HandlersRunner"

    def run(self, output: AfterEvaluationOutput):
        for func in self._handlers.values():
            result = func(output)
            if result.get(PhlowerHandlerRegisteredKey.TERMINATE):
                self._terminate = True
                break

    @property
    def terminate_training(self) -> bool:
        return self._terminate

    def state_dict(self) -> dict[str, dict]:
        items = {
            name: func.state_dict() for name, func in self._handlers.items()
        }
        return {"handlers": items}

    def load_state_dict(self, state_dict: Mapping) -> None:
        assert isinstance(state_dict["handlers"], dict)
        for k, v in state_dict["handlers"].items():
            if k not in self._handlers:
                raise ValueError(
                    f"Handler named {k} is not initialized. "
                    "Please check handler setting in trainer."
                )
            self._handlers[k].load_state_dict(v)
