import abc
import pathlib

from torch.utils.data import Dataset

from phlower.base.array import IPhlowerArray
from phlower.data._lumped_data import LumpedArrayData
from phlower.io import PhlowerDirectory


class IPhlowerDataset(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __getitem__(self, idx: int) -> LumpedArrayData:
        raise NotImplementedError()


class LazyPhlowerDataset(Dataset):
    """This Dataset reads input file every time to get data"""

    def __init__(
        self,
        x_variable_names: list[str],
        y_variable_names: list[str] | None,
        directories: list[pathlib.Path],
        *,
        support_names: list[str] = None,
        num_workers: int = 0,
        allow_no_y_data: bool = False,
        decrypt_key: bytes | None = None,
        **kwargs
    ):

        self._x_variable_names = x_variable_names
        self._y_varaible_names = y_variable_names
        self._directories: list[PhlowerDirectory] = [
            PhlowerDirectory(d) for d in directories
        ]
        self._support_names = support_names if support_names is not None else []

        self._num_workers = num_workers
        self._allow_no_y_data = allow_no_y_data
        self._decrypt_key = decrypt_key

    def __len__(self):
        return len(self._directories)

    def __getitem__(self, idx: int) -> LumpedArrayData:
        data_directory = self._directories[idx]
        x_data = self._load_data(
            data_directory, self._x_variable_names, allow_missing=False
        )
        y_data = self._load_data(
            data_directory,
            self._y_varaible_names,
            allow_missing=self._allow_no_y_data,
        )
        support_data = self._load_data(
            data_directory, self._support_names, allow_missing=False
        )
        return LumpedArrayData(
            x_data=x_data,
            y_data=y_data,
            sparse_supports=support_data,
            data_directory=data_directory,
        )

    def _load_data(
        self,
        data_directory: PhlowerDirectory,
        variable_names: list[str],
        allow_missing: bool,
    ) -> dict[str, IPhlowerArray]:
        variable_files = [
            (
                name,
                data_directory.find_variable_file(
                    name, allow_missing=allow_missing
                ),
            )
            for name in variable_names
        ]
        return {
            name: phlower_file.load(decrypt_key=self._decrypt_key)
            for name, phlower_file in variable_files
        }
