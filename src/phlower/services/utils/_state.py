import pathlib
from typing import NamedTuple

from phlower.settings import PhlowerSetting


class TrainingTerminatedState(NamedTuple):
    output_directory: pathlib.Path
    setting: PhlowerSetting
    exception: Exception
    traceback_info: str
