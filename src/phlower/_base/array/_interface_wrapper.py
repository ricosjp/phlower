from __future__ import annotations

import abc
from collections.abc import Callable

import torch

from phlower._base import PhysicalDimensions
from phlower.utils.typing import ArrayDataType


class IPhlowerArray(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(
        self, data: ArrayDataType, dimensions: PhysicalDimensions
    ) -> None: ...

    @property
    @abc.abstractmethod
    def dimension(self) -> PhysicalDimensions | None: ...

    @property
    @abc.abstractmethod
    def size(self) -> int: ...

    @property
    @abc.abstractmethod
    def is_time_series(self) -> bool: ...

    @property
    @abc.abstractmethod
    def is_sparse(self) -> bool: ...

    @property
    @abc.abstractmethod
    def is_voxel(self) -> bool: ...

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int]: ...

    @abc.abstractmethod
    def apply(
        self,
        function: Callable[[ArrayDataType], ArrayDataType],
        componentwise: bool,
        *,
        skip_nan: bool = False,
        use_diagonal: bool = False,
        **kwards,
    ) -> IPhlowerArray:
        """Apply user defined function

        Parameters
        ----------
        function : Callable[[T], T]
            function to apply
        componentwise : bool
            If True, fucnction is applied by component wise way
        skip_nan : bool, optional
            If True, np.nan value is ignored. This option is valid
             only when data is np.ndarray. By default False
        use_diagonal : bool, optional
            If True, only diagonal values are used. This option is valid
             only when data is sparse array. By default False

        Returns
        -------
        T: Same type of instance caraible

        """
        ...

    @abc.abstractmethod
    def reshape(
        self,
        componentwise: bool,
        *,
        skip_nan: bool = False,
        use_diagonal: bool = False,
        **kwrds,
    ) -> IPhlowerArray: ...

    @abc.abstractmethod
    def to_tensor(
        self,
        device: str | torch.device | None = None,
        non_blocking: bool = False,
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def to_numpy(self) -> ArrayDataType: ...

    @abc.abstractmethod
    def slice_along_time_axis(self, time_slice: slice) -> ArrayDataType: ...
