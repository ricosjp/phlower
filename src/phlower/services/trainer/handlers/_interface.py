import abc
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from phlower.services.trainer._pass_items import AfterEvaluationOutput


class IHandlerCall(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def name(cls) -> str: ...

    @abc.abstractmethod
    def __call__(self, output: AfterEvaluationOutput) -> dict[str, Any]:
        """Run handler's program. If output dictionary conatins "TERMINATE" and
         its value is True, training process is terminated forcefully.

        Args:
            output (AfterEvaluationOutput): output data from evaluation runner

        Returns:
            dict[str, Any]: output data
        """
        ...

    @abc.abstractmethod
    def state_dict(self) -> OrderedDict[str, float]: ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Mapping) -> None: ...
