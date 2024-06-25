import pathlib

from phlower.io._files import (
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
    def numpy_file(file_path: pathlib.Path | str) -> IPhlowerNumpyFile:
        return PhlowerNumpyFile(file_path)

    @staticmethod
    def pickle_file(file_path: pathlib.Path | str) -> IPhlowerPickleFile:
        return PhlowerPickleFile(file_path)

    @staticmethod
    def checkpoint_file(
        file_path: pathlib.Path | str,
    ) -> IPhlowerCheckpointFile:
        return PhlowerCheckpointFile(file_path)

    @staticmethod
    def yaml_file(file_path: pathlib.Path | str) -> IPhlowerYamlFile:
        return PhlowerYamlFile(file_path)
