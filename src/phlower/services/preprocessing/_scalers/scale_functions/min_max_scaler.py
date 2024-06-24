from sklearn import preprocessing

from phlower.services.preprocessing._scalers import IPhlowerScaler


class MinMaxScaler(preprocessing.MinMaxScaler, IPhlowerScaler):
    @classmethod
    def create(cls, name: str, **kwards):
        if name == "min_max":
            return MinMaxScaler(**kwards)

        raise NotImplementedError() 

    @classmethod
    def get_registered_names(self) -> list[str]:
        return ["min_max"]

    def __init__(
        self,
        feature_range=...,
        *,
        copy=True,
        clip=False,
        **kwargs
    ):
        super().__init__(feature_range, copy=copy, clip=clip)

    @property
    def use_diagonal(self) -> bool:
        return False

    def is_erroneous(self) -> bool:
        return False
