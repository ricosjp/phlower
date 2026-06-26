from collections.abc import Callable

import torch


def _max_pool(
    tensors: list[torch.Tensor], pooling_dimension: int
) -> torch.Tensor:
    return torch.stack(
        [
            torch.max(t, dim=pooling_dimension, keepdim=False)[0]
            for t in tensors
        ],
        dim=pooling_dimension,
    )


def _mean(tensors: list[torch.Tensor], pooling_dimension: int) -> torch.Tensor:
    return torch.stack(
        [torch.mean(t, dim=pooling_dimension, keepdim=False) for t in tensors],
        dim=pooling_dimension,
    )


_PoolOperator = Callable[[list[torch.Tensor], int], torch.Tensor]


class PoolingSelector:
    """PoolingSelector is a class that selects the pooling operator
    based on the name.

    Parameters
    ----------
    name: str
        Name of the pooling operator. "max" or "mean".
        Default is "max".

    """

    _REGISTERED_POOL_OP: dict[str, _PoolOperator] = {
        "max": _max_pool,
        "mean": _mean,
    }

    @staticmethod
    def is_exist(name: str) -> bool:
        """Check if the pooling operator exists.

        Parameters
        ----------
        name: str
            Name of the pooling operator. "max" or "mean".
            Default is "max".

        Returns
        -------
        bool
            True if the pooling operator exists, False otherwise.
        """
        return name in PoolingSelector._REGISTERED_POOL_OP

    @staticmethod
    def select(name: str) -> _PoolOperator:
        """Select the pooling operator based on the name.

        Parameters
        ----------
        name: str
            Name of the pooling operator. "max" or "mean".
            Default is "max".

        Returns
        -------
        Callable[[torch.Tensor], torch.Tensor]
            Pooling operator.
        """
        if name not in PoolingSelector._REGISTERED_POOL_OP:
            raise ValueError(f"Pooling operator {name} is not registered.")

        return PoolingSelector._REGISTERED_POOL_OP[name]
