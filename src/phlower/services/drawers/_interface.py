import abc
import pathlib

from phlower.nn._interface_module import IPhlowerModuleAdapter


class IPhlowerDrawer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def output(
        self, modules: list[IPhlowerModuleAdapter], file_path: pathlib.Path
    ) -> None: ...
