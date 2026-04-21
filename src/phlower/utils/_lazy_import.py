import importlib
import types
from typing import Any

# This module provides a mechanism for lazy importing of modules.
# The following items are referenced for the implementation:
# * https://docs.python.org/3/reference/datamodel.html#module-objects
# * https://github.com/optuna/optuna/blob/809370999e4fa636e134d4f23ea3ebb96fd1d730/optuna/_imports.py#L111


class _LazyImport(types.ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._name = name

    def _load(self) -> types.ModuleType:
        module = importlib.import_module(self._name)
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item: str) -> Any:  # noqa: ANN401
        return getattr(self._load(), item)
