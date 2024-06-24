from sklearn.base import BaseEstimator, TransformerMixin

from phlower.services.preprocessing._scalers import  IPhlowerScaler


class IdentityScaler(BaseEstimator, TransformerMixin, IPhlowerScaler):
    """Class to perform identity conversion (do nothing)."""
    @classmethod
    def create(cls, name: str, **kwards):
        if name == "identity":
            return IdentityScaler(**kwards)

    @classmethod
    def get_registered_names(cls) -> list[str]:
        return ["identity"]

    def __init__(self, **kwards) -> None:
        super().__init__()

    @property
    def use_diagonal(self) -> bool:
        return False

    def is_erroneous(self) -> bool:
        return False

    def partial_fit(self, data):
        return

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data
