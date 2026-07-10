from __future__ import annotations

import abc

from phlower.services.utils import TrainingTerminatedState
from phlower.utils import get_logger

_logger = get_logger(__name__)


class IPhlowerErrorHandler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def should_suppress(
        self, terminated_state: TrainingTerminatedState
    ) -> bool | None:
        """
        Receives a TrainingTerminatedState object and handles
        the exception accordingly.

        If the handler decides to allow the training to continue,
        it should return True.
        If at least one handler in the chain returns True,
        the training will continue.
        """
        ...


class DumpTracebackHandler(IPhlowerErrorHandler):
    def should_suppress(
        self, terminated_state: TrainingTerminatedState
    ) -> bool | None:

        log_path = terminated_state.output_directory / "training_error.txt"
        with log_path.open("a") as f:
            f.write(str(terminated_state.exception) + "\n")
            f.write("Traceback:\n")
            f.write(terminated_state.traceback_info + "\n")
            f.write("-" * 50 + "\n")


class AllowWhenOneEpochCompletedHandler(IPhlowerErrorHandler):
    def should_suppress(
        self, terminated_state: TrainingTerminatedState
    ) -> bool | None:
        output_directory = terminated_state.output_directory

        pth_files = list((output_directory / "weights").glob("*.pth*"))
        if len(pth_files) == 0:
            _logger.info(
                f"No .pth files found in {output_directory / 'weights'}. "
                "Training is terminated before one epoch is completed. "
                "Raising the exception."
            )
            return

        _logger.info(
            "Training is terminated after one epoch is completed. "
            "Allowing the exception to be ignored."
        )
        return True


class ErrorHandlerChain(IPhlowerErrorHandler):
    def __init__(self, handlers: list[str]):
        self._handlers = [
            PhlowerErrorHandlerFactory.create(handler_name)
            for handler_name in handlers
        ]

    def should_suppress(
        self, terminated_state: TrainingTerminatedState
    ) -> bool | None:
        if len(self._handlers) == 0:
            return None

        allows = [
            handler.should_suppress(terminated_state)
            for handler in self._handlers
        ]
        allows = [allow for allow in allows if allow is not None]
        if any(allows):
            return True

        return None


class PhlowerErrorHandlerFactory:
    _REGISTERED: dict[str, type[IPhlowerErrorHandler]] = {
        "allow_when_one_epoch_completed": AllowWhenOneEpochCompletedHandler,
        "dump_traceback": DumpTracebackHandler,
    }

    @classmethod
    def register(
        cls,
        name: str,
        handler_type: type[IPhlowerErrorHandler],
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
    def create(
        cls, name: str, params: dict | None = None
    ) -> IPhlowerErrorHandler:
        params = params or {}
        if name not in cls._REGISTERED:
            raise NotImplementedError(f"Handler {name} is not registered.")
        return cls._REGISTERED[name](**params)
