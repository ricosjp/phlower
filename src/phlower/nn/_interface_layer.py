import abc

from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections


class IPhlowerLayer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor]
    ) -> IPhlowerTensorCollections:
        raise NotImplementedError()
