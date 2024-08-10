from collections.abc import Callable

import torch

from phlower._base.tensors import PhlowerTensor
from phlower.nn._core_modules import _functions


class ExtendedLinearList(torch.nn.Module):
    def __init__(
        self,
        nodes: list[int],
        activations: list[str],
        bias: bool,
        dropouts: list[float] | None = None,
    ) -> None:
        super().__init__()

        self._nodes = nodes
        self._activations = activations
        if dropouts is None:
            dropouts = []
        self._dropouts = dropouts
        self._validate_args()

        self._n_chains = len(self._nodes)
        self._linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(n1, n2, bias=bias)
                for n1, n2 in zip(
                    self._nodes[:-1], self._nodes[1:], strict=False
                )
            ]
        )
        self._activators = [
            ActivationSelector.select(name) for name in self._activations
        ]

        # for m in self._linears:
        #     torch.nn.init.xavier_uniform_(m.weight.data)

    def __len__(self) -> int:
        return len(self._linears)

    def forward_part(self, x: PhlowerTensor, *, index: int) -> PhlowerTensor:
        assert index < self._n_chains

        h = self._linears[index](x)
        h = torch.nn.functional.dropout(
            h, p=self._dropouts[index], training=self.training
        )
        h = self._activators[index](h)
        return h

    def forward(self, x: PhlowerTensor) -> PhlowerTensor:
        h = x
        for i in range(self._n_chains - 1):
            h = self.forward_part(h, index=i)
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


class ActivationSelector:
    _REGISTERED_ACTIVATIONS = {
        "identity": _functions.identity,
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "sqrt": torch.sqrt,
        "tanh": torch.tanh,
    }

    @staticmethod
    def select(name: str | None) -> Callable[[torch.Tensor], torch.Tensor]:
        if name is None:
            name = "identity"
        return ActivationSelector._REGISTERED_ACTIVATIONS[name]

    @staticmethod
    def is_exists(name: str) -> bool:
        return name in ActivationSelector._REGISTERED_ACTIVATIONS
