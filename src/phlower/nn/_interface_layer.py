import abc

from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.base.tensors import PhlowerTensor


class IPhlowerLayer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor]
    ) -> IPhlowerTensorCollections:
        raise NotImplementedError()
