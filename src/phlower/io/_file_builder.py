import pathlib

from phlower.io._files import (
    IPhlowerCheckpointFile,
    IPhlowerNumpyFile,
    IPhlowerYamlFile,
    PhlowerCheckpointFile,
    PhlowerNumpyFile,
    PhlowerYamlFile,
)


class PhlowerFileBuilder:
    @staticmethod
    def numpy_file(
        file_path: pathlib.Path | str | PhlowerNumpyFile,
    ) -> IPhlowerNumpyFile:
        if isinstance(file_path, IPhlowerNumpyFile):
            return PhlowerNumpyFile(file_path.file_path)

        return PhlowerNumpyFile(file_path)

    @staticmethod
    def checkpoint_file(
        file_path: pathlib.Path | str,
    ) -> IPhlowerCheckpointFile:
        return PhlowerCheckpointFile(file_path)

    @staticmethod
    def yaml_file(file_path: pathlib.Path | str) -> IPhlowerYamlFile:
        return PhlowerYamlFile(file_path)
