from typing import Callable

import torch

from phlower._base.tensors import PhlowerTensor


class ExtendedLinearList(torch.nn.Module):

    def __init__(
        self,
        nodes: list[int],
        activations: list[str],
        dropouts: list[float],
        bias: bool,
    ) -> None:
        super().__init__()

        self._nodes = nodes
        self._activations = activations
        self._dropouts = dropouts
        self._validate_args()

        self._n_chains = len(self._nodes)
        self._linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(n1, n2, bias=bias)
                for n1, n2 in zip(self._nodes[:-1], self._nodes[1:])
            ]
        )
        self._activators = [
            ActivationSelector.select(name) for name in self._activations
        ]

    def __len__(self) -> int:
        return len(self._linears)

    def forward(self, x: PhlowerTensor, *, index: int) -> PhlowerTensor:
        assert index < self._n_chains

        h = self._linears[index].forward(x)
        h = torch.nn.functional.dropout(
            h, p=self._dropouts[index], training=self.training
        )
        h = self._activators[index](h)
        return h

    def _validate_args(self) -> None:
        assert len(self._nodes) >= 2

        if len(self._activations) == 0:
            self._activations = [
                "identity" for _ in range(len(self._nodes) - 1)
            ]
        assert len(self._nodes) == len(self._activations) + 1

        if len(self._dropouts) == 0:
            self._dropouts = [0 for _ in range(len(self._nodes) - 1)]
        for v in self._dropouts:
            assert v < 1.0
        assert len(self._nodes) == len(self._dropouts) + 1


def identity(x):
    return x


class ActivationSelector:
    _REGISTERED_ACTIVATIONS = {
        "identity": identity,
        "relu": torch.relu,
        "tanh": torch.tanh,
    }

    @staticmethod
    def select(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        return ActivationSelector._REGISTERED_ACTIVATIONS[name]
