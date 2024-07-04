from sklearn import preprocessing

from phlower.services.preprocessing._scalers import IPhlowerScaler
from phlower.utils.enums import PhlowerScalerName


class MinMaxScaler(preprocessing.MinMaxScaler, IPhlowerScaler):
    @classmethod
    def create(cls, name: str, **kwards):
        if name == PhlowerScalerName.MIN_MAX.value:
            return MinMaxScaler(**kwards)

        raise NotImplementedError()

    @classmethod
    def get_registered_names(self) -> list[str]:
        return [PhlowerScalerName.MIN_MAX.value]

    def __init__(self, feature_range=..., *, copy=True, clip=False, **kwargs):
        super().__init__(feature_range, copy=copy, clip=clip)

    @property
    def use_diagonal(self) -> bool:
        return False

    def is_erroneous(self) -> bool:
        return False
