import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

from phlower.services.preprocessing._scalers import IPhlowerScaler
from phlower.utils.enums import PhlowerScalerName


class IsoAMScaler(BaseEstimator, TransformerMixin, IPhlowerScaler):
    """Class to perform scaling for IsoAM based on
    https://arxiv.org/abs/2005.06316.
    """

    @classmethod
    def create(cls, name: str, **kwards):
        if name == PhlowerScalerName.ISOAM_SCALE.name:
            return IsoAMScaler(**kwards)

        raise NotImplementedError()

    @classmethod
    def get_registered_names(cls) -> list[str]:
        return [PhlowerScalerName.ISOAM_SCALE.name]

    def __init__(self, other_components: list[str], **kwargs):
        self.var_: float = 0.0
        self.std_: float = 0.0
        self.mean_square_: float = 0.0
        self.n_: int = 0
        self.other_components = other_components

        if len(self.other_components) == 0:
            raise ValueError(
                "To use IsoAMScaler, feed other_components: "
                f"{self.other_components}"
            )
        return

    @property
    def component_dim(self) -> int:
        return len(self.other_components) + 1

    @property
    def use_diagonal(self) -> bool:
        return True

    def is_erroneous(self) -> bool:
        return np.any(np.isnan(self.var_))

    def partial_fit(self, data):
        self._update(data)
        return self

    def _update(self, diagonal_data: np.ndarray):
        if len(diagonal_data.shape) != 1:
            raise ValueError(f"Input data should be 1D: {diagonal_data}")
        m = len(diagonal_data)
        mean_square = (
            self.mean_square_ * self.n_ + np.sum(diagonal_data**2)
        ) / (self.n_ + m)

        self.mean_square_ = mean_square
        self.n_ += m

        # To use mean_i [x_i^2 + y_i^2 + z_i^2], multiply by the dim
        self.var_ = self.mean_square_ * self.component_dim
        self.std_ = np.sqrt(self.var_)
        return

    def _raise_if_not_sparse(self, data):
        if not sp.issparse(data):
            raise ValueError("Data is not sparse")
        return

    def transform(self, data):
        if self.std_ == 0.0:
            scale = 0.0
        else:
            scale = 1 / self.std_
        return data * scale

    def inverse_transform(self, data):
        inverse_scale = self.std_
        return data * inverse_scale
