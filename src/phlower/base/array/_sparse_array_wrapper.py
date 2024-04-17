import warnings
from typing import Callable

import torch

from phlower.base.array._interface_wrapper import IPhlowerArray
from phlower.base.tensors import PhlowerTensor, phlower_tensor
from phlower.utils.typing import SparseArrayType


class SparseArrayWrapper(IPhlowerArray):
    def __init__(self, data: SparseArrayType) -> None:
        self._sparse_data = data

    @property
    def shape(self) -> tuple[int]:
        return self._sparse_data.shape

    def apply(
        self,
        apply_function: Callable[[SparseArrayType], SparseArrayType],
        componentwise: bool,
        *,
        use_diagonal: bool = False,
        **kwards
    ) -> SparseArrayType:

        if use_diagonal:
            raise ValueError(
                "Cannot set use_diagonal=True in self.apply function. "
                "use_diagonal is only allows in self.reshape"
            )

        reshaped_array = self.reshape(
            componentwise=componentwise, use_diagonal=use_diagonal
        )

        result = apply_function(reshaped_array)
        return result.reshape(self.shape).tocoo()

    def reshape(
        self, componentwise: bool, *, use_diagonal: bool = False, **kwards
    ) -> SparseArrayType:
        if componentwise and use_diagonal:
            warnings.warn(
                "component_wise and use_diagonal cannnot use"
                " at the same time. use_diagonal is ignored"
            )

        if componentwise:
            return self._sparse_data

        if use_diagonal:
            reshaped = self._sparse_data.diagonal()
            return reshaped

        reshaped = self._sparse_data.reshape((self.shape[0] * self.shape[1], 1))
        return reshaped

    def to_phlower_tensor(
        self,
        device: str | torch.device | None,
        non_blocking: bool = False,
        dimension: dict[str, float] | None = None,
    ) -> PhlowerTensor:
        sparse_tensor = torch.sparse_coo_tensor(
            torch.stack(
                [
                    torch.LongTensor(self._sparse_data.row),
                    torch.LongTensor(self._sparse_data.col),
                ]
            ),
            torch.from_numpy(self._sparse_data.data),
            self._sparse_data.shape,
        )
        _tensor = phlower_tensor(tensor=sparse_tensor, dimension=dimension)
        _tensor.to(device=device, non_blocking=non_blocking)
        return _tensor

    def numpy(self) -> SparseArrayType:
        return self._sparse_data
