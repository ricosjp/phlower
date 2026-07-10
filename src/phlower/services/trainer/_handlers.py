from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, overload

from phlower.services.trainer._handler_functions import (
    EarlyStopping,
    NaNStoppingHandler,
)
from phlower.settings import PhlowerTrainerSetting
from phlower.utils.enums import (
    PhlowerHandlerTrigger,
)
from phlower.utils.exceptions import PhlowerHandlerStopTraining
from phlower.utils.typing import (
    AfterEvaluationOutput,
    IPhlowerHandler,
    PhlowerHandlerAnalysisResult,
)


class PhlowerHandlersFactory:
    _REGISTERED: dict[str, type[IPhlowerHandler]] = {
        "EarlyStopping": EarlyStopping,
        "NaNStoppingHandler": NaNStoppingHandler,
    }

    @classmethod
    def register(
        cls,
        name: str,
        handler_type: type[IPhlowerHandler],
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
    def create(cls, name: str, params: dict) -> IPhlowerHandler:
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
        self._result = PhlowerHandlerAnalysisResult(False, None)
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

    @overload
    def run(
        self,
        output: AfterEvaluationOutput,
        trigger: Literal[PhlowerHandlerTrigger.epoch_completed],
    ) -> None: ...

    @overload
    def run(
        self,
        output: float,
        trigger: Literal[PhlowerHandlerTrigger.iteration_completed],
    ) -> None: ...

    def update_activity(self, continue_count: int):
        for func in self._handlers.values():
            func.update_activity(continue_count)

    def run(
        self,
        output: AfterEvaluationOutput | float,
        trigger: PhlowerHandlerTrigger,
    ):
        for func in self._handlers.values():
            if func.trigger() != trigger:
                continue

            result = func(output)
            if result.terminate_training:
                # TODO: Consider multiple reasons
                self._result = result
                break
        if self._result.terminate_training:
            raise PhlowerHandlerStopTraining(self._result.reason)

    def attach(
        self,
        name: str,
        handler: IPhlowerHandler,
        allow_overwrite: bool = False,
    ) -> None:
        if (not allow_overwrite) and (name in self._handlers):
            raise ValueError(f"Handler named {name} is already attached.")
        self._handlers[name] = handler

    @property
    def terminate_training(self) -> bool:
        return self._result.terminate_training

    @property
    def handler_result(self) -> PhlowerHandlerAnalysisResult:
        return self._result

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
