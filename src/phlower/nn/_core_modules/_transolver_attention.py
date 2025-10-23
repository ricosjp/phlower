from __future__ import annotations

import einops
import torch
from typing_extensions import Self

from phlower._base._functionals import unbatch
from phlower._base.tensors import (
    PhlowerTensor,
)
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._functionals import _functions
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import TransolverAttentionSetting
from phlower.utils import get_logger

_logger = get_logger(__name__)


class TransolverAttention(IPhlowerCoreModule, torch.nn.Module):
    """TransolverAttention is a neural network module that performs
    physics-attention on the input tensor, as used in Transolver.

    It performs attention in three steps:
    1. Slice: Project features into physics-aware/slice tokens.
    2. Attention: Compute attention among physics-aware/slice tokens.
    3. Deslice: Project physics-aware/slice tokens back to original space.

    Ref: https://arxiv.org/abs/2402.02366

    Parameters
    ----------
    nodes: list[int]
        List of feature dimension sizes (The last value of tensor shape).
        [input_dim, inner_dim, output_dim]
    heads: int
        Number of attention heads. Default is 8.
    slice_num: int
        Number of slice tokens. Default is 32.
    dropout: float
        Dropout rate. Default is 0.0.
    unbatch_key: str | None
        Key of the unbatch operation.

    Examples
    --------
    >>> attention = TransolverAttention(
    ...     nodes=[256, 256, 256],
    ...     heads=8,
    ...     slice_num=32,
    ...     dropout=0.0,
    ... )
    >>> attention(data)
    """

    @classmethod
    def from_setting(cls, setting: TransolverAttentionSetting) -> Self:
        """Create TransolverAttention from setting object

        Args:
            setting (TransolverAttentionSetting): setting object

        Returns:
            Self: TransolverAttention
        """
        return cls(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of TransolverAttention

        Returns:
            str: name
        """
        return "TransolverAttention"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        nodes: list[int],
        heads: int = 8,
        slice_num: int = 32,
        dropout: float = 0.0,
        unbatch_key: str | None = None,
    ):
        super().__init__()

        self._nodes = nodes
        self._heads = heads
        self._slice_num = slice_num
        self._dropout_rate = dropout
        self._reference_batch_name = unbatch_key

        in_dim, inner_dim, out_dim = nodes
        assert inner_dim % heads == 0
        dim_head = inner_dim // heads

        # Temperature parameter for slicing
        self._temperature = torch.nn.Parameter(
            torch.ones([1, heads, 1, 1]) * 0.5
        )

        # Projections for point tokens
        self._to_k_x = torch.nn.Linear(in_dim, inner_dim)
        self._to_v_x = torch.nn.Linear(in_dim, inner_dim)
        self._to_slice_weights = torch.nn.Linear(dim_head, slice_num)

        # Projections for slice tokens
        self._to_q_s = torch.nn.Linear(dim_head, dim_head, bias=False)
        self._to_k_s = torch.nn.Linear(dim_head, dim_head, bias=False)
        self._to_v_s = torch.nn.Linear(dim_head, dim_head, bias=False)

        # Output projection
        self._to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, out_dim),
            torch.nn.Dropout(dropout),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        torch.nn.init.orthogonal_(self._to_slice_weights.weight)

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None: ...

    def get_reference_name(self) -> str | None:
        return None

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None = None,
        **kwards,
    ) -> PhlowerTensor:
        """forward function which overloads torch.nn.Module

        Args:
            data: IPhlowerTensorCollections
                data which receives from predecessors
            field_data: ISimulationField | None
                Constant information through training or prediction

        Returns:
            PhlowerTensor: Tensor object
        """
        x = data.unique_item()
        node_dim = x.shape_pattern.nodes_dim
        unbatched_x = self._unbatch(x, field_data=field_data)

        x = torch.cat([self._forward(x) for x in unbatched_x], dim=node_dim)

        return x

    def _unbatch(
        self, x: PhlowerTensor, field_data: ISimulationField | None
    ) -> list[PhlowerTensor]:
        if (field_data is None) or (self._reference_batch_name is None):
            _logger.info(
                "batch info is not passed to TransolverAttention. "
                "Unbatch operation is skipped."
            )
            return [x]

        return unbatch(
            x,
            n_nodes=field_data.get_batched_n_nodes(self._reference_batch_name),
        )

    def _forward(self, x_ph: PhlowerTensor) -> PhlowerTensor:
        x, restore_info = _functions.to_batch_node_feature(x_ph)

        ### (1) Slice
        # Project to keys
        k_x = self._to_k_x(x)
        k_x = einops.rearrange(k_x, "b n (h f) -> b h n f", h=self._heads)

        # Project to values
        v_x = self._to_v_x(x)
        v_x = einops.rearrange(v_x, "b n (h f) -> b h n f", h=self._heads)

        # Soft-assign each point to slices
        slice_weights = self._to_slice_weights(k_x)  # B H N S (S = slice_num)
        slice_weights = torch.nn.functional.softmax(
            slice_weights / self._temperature, dim=-1
        )

        # Aggregate points to slices
        slice_token = slice_weights.transpose(-2, -1) @ v_x  # B H S F

        # Normalize slice tokens
        slice_norm = slice_weights.sum(-2)  # B H S
        slice_token = slice_token / (slice_norm.unsqueeze(-1) + 1e-5)

        ### (2) Attention among slice tokens
        q_s = self._to_q_s(slice_token)
        k_s = self._to_k_s(slice_token)
        v_s = self._to_v_s(slice_token)

        out_slice_token = torch.nn.functional.scaled_dot_product_attention(
            q_s,
            k_s,
            v_s,
            dropout_p=self._dropout_rate,
        )

        ### (3) Deslice
        out_x = slice_weights @ out_slice_token  # B H N F
        out_x = einops.rearrange(out_x, "b h n f -> b n (h f)")
        out_x = self._to_out(out_x)

        out_x_ph = _functions.from_batch_node_feature(out_x, restore_info)
        return out_x_ph
