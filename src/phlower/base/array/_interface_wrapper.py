import abc
from typing import Callable

import torch

from phlower.base.tensors import PhlowerTensor
from phlower.utils.typing import ArrayDataType


class IPhlowerArray(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, data: ArrayDataType) -> None:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int]:
        raise NotImplementedError()

    @abc.abstractmethod
    def apply(
        self,
        function: Callable[[ArrayDataType], ArrayDataType],
        componentwise: bool,
        *,
        skip_nan: bool = False,
        use_diagonal: bool = False,
        **kwards
    ) -> ArrayDataType:
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
        raise NotImplementedError()

    @abc.abstractmethod
    def reshape(
        self,
        componentwise: bool,
        *,
        skip_nan: bool = False,
        use_diagonal: bool = False,
        **kwrds
    ) -> ArrayDataType:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_phlower_tensor(
        self,
        device: str | torch.device | None,
        non_blocking: bool = False,
        dimension: dict[str, float] | None = None,
    ) -> PhlowerTensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def numpy(self) -> ArrayDataType:
        raise NotImplementedError()
