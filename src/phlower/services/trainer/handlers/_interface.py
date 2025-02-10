import abc
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from phlower.services.trainer._pass_items import AfterEpochRunnerOutput


class IHandlerCall(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    @classmethod
    def name(cls) -> str: ...

    @abc.abstractmethod
    def __call__(self, output: AfterEpochRunnerOutput) -> dict[str, Any]: ...

    @abc.abstractmethod
    def state_dict(self) -> OrderedDict[str, float]: ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Mapping) -> None: ...
