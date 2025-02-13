from __future__ import annotations

from collections.abc import Mapping

from phlower.services.trainer._handler_functions import create_handler
from phlower.services.trainer._pass_items import AfterEvaluationOutput
from phlower.settings import PhlowerTrainerSetting
from phlower.utils.typing import PhlowerHandlerType
from phlower.utils.enums import PhlowerHandlerRegisteredKey



class HandlersRunner(PhlowerHandlerType):
    @classmethod
    def from_setting(
        cls,
        setting: PhlowerTrainerSetting,
        user_defined_handlers: dict[str, PhlowerHandlerType] | None = None,
    ) -> HandlersRunner:
        return HandlersRunner(
            handlers_params={
                v.handler: v.parameters for v in setting.handler_settings
            },
            user_defined_handlers=user_defined_handlers,
        )

    def __init__(
        self,
        handlers_params: dict[str, dict],
        user_defined_handlers: dict[str, PhlowerHandlerType] | None = None,
    ):
        self._terminate = False
        self._handlers = {
            name: create_handler(name, params)
            for name, params in handlers_params.items()
        }

        user_defined_handlers = user_defined_handlers or {}
        for name, u_handler in user_defined_handlers.items():
            if name in self._handlers:
                raise ValueError(f"{name} has already been registered.")
            self._handlers[name] = u_handler

    @property
    def n_handlers(self) -> int:
        return len(self._handlers)

    @classmethod
    def name(cls) -> str:
        return "HandlersRunner"

    def __call__(self, output: AfterEvaluationOutput):
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
