from __future__ import annotations

from collections.abc import Mapping

from phlower.services.trainer._pass_items import AfterEvaluationOutput
from phlower.services.trainer.handlers import IHandlerCall, create_handler
from phlower.settings import PhlowerTrainerSetting


class HandlersRunner(IHandlerCall):
    @classmethod
    def from_setting(cls, setting: PhlowerTrainerSetting) -> HandlersRunner:
        return HandlersRunner(
            handlers={v.handler: v.parameters for v in setting.handler_setting}
        )

    def __init__(self, handlers: dict[str, dict]):
        self._terminate = False
        self._handlers = {
            name: create_handler(name, params)
            for name, params in handlers.items()
        }

    @classmethod
    def name(cls) -> str:
        return "HandlersRunner"

    def __call__(self, output: AfterEvaluationOutput):
        for func in self._handlers.values():
            result = func(output)
            if result.get("TERMINATE"):
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
        for k, v in state_dict["handlers"]:
            self._handlers[k].load_state_dict(v[k])
