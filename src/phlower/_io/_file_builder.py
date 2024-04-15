import pathlib

from ._files import (
    IPhlowerCheckpointFile,
    IPhlowerNumpyFile,
    IPhlowerPickleFile,
    IPhlowerYamlFile,
    PhlowerCheckpointFile,
    PhlowerNumpyFile,
    PhlowerPickleFile,
    PhlowerYamlFile,
)


class PhlowerFileBuilder:
    @staticmethod
    def numpy_file(file_path: pathlib.Path) -> IPhlowerNumpyFile:
        return PhlowerNumpyFile(file_path)

    @staticmethod
    def pickle_file(file_path: pathlib.Path) -> IPhlowerPickleFile:
        return PhlowerPickleFile(file_path)

    @staticmethod
    def checkpoint_file(file_path: pathlib.Path) -> IPhlowerCheckpointFile:
        return PhlowerCheckpointFile(file_path)

    @staticmethod
    def yaml_file(file_path: pathlib.Path) -> IPhlowerYamlFile:
        return PhlowerYamlFile(file_path)
