from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from typing import Literal

import torch

import phlower
from phlower import ISimulationField
from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _functions, _utils
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings._isogcn_setting import (
    IsoGCNNeumannLinearType,
    IsoGCNPropagationType,
    IsoGCNSetting,
)


class IsoGCN(IPhlowerCoreModule, torch.nn.Module):
    """Graph Convolutional Neural Network"""

    @classmethod
    def from_setting(cls, setting: IsoGCNSetting) -> IsoGCN:
        """Create IsoGCN from IsoGCNSetting instance

        Args:
            setting (GCNSetting): setting object for IsoGCN

        Returns:
            IsoGCN: IsoGCN object
        """
        return IsoGCN(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "IsoGCN"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        nodes: list[int],
        support_names: list[str],
        propagations: list[IsoGCNPropagationType],
        subchain_activations: list[str],
        subchain_dropouts: list[str],
        subchain_bias: bool,
        coefficient_activations: list[str],
        coefficient_dropouts: list[str],
        coefficient_bias: bool,
        mul_order: Literal["ah_w", "a_hw"] = "ah_w",
        neumann_active: bool = False,
        neumann_linear_model_type: IsoGCNNeumannLinearType = (
            IsoGCNNeumannLinearType.reuse_graph_weight
        ),
        neumann_factor: float = 1.0,
        neumann_apply_sigmoid_ratio: bool = False,
        inversed_moment_name: str = "",
    ) -> None:
        super().__init__()

        self._weight = _utils.ExtendedLinearList(
            nodes=[nodes[0], nodes[-1]],
            activations=subchain_activations,
            dropouts=subchain_dropouts,
            bias=subchain_bias,
        )
        self._use_coefficient = True
        self._coefficient_network = _utils.ExtendedLinearList(
            nodes=nodes,
            activations=coefficient_activations,
            dropouts=coefficient_dropouts,
            bias=coefficient_bias,
        )
        self._propagations = propagations
        self._nodes = nodes
        self._support_names = support_names
        self._mul_order = mul_order

        self._inversed_moment_name = inversed_moment_name
        self._neumann_active = neumann_active
        self._neumann_linear_type = neumann_linear_model_type
        self._neumann_factor = neumann_factor
        self._neumann_apply_sigmoid_ratio = neumann_apply_sigmoid_ratio
        self._neumann_layer = self._create_neumann_layer()

    def _create_neumann_layer(self) -> Callable[[PhlowerTensor], PhlowerTensor]:
        if self._neumann_linear_type == IsoGCNNeumannLinearType.identity:
            return phlower.nn.functions.identity

        if self._neumann_linear_type == IsoGCNNeumannLinearType.create_new:
            return torch.nn.Linear(*self._weight[0].weight.shape, bias=False)

        if (
            self._neumann_linear_type
            == IsoGCNNeumannLinearType.reuse_graph_weight
        ):
            return self._weight[0]

        raise NotImplementedError(
            f"{self._neumann_linear_type} is not implemented."
        )

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None: ...

    def get_reference_name(self) -> str | None:
        return None

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField,
        **kwards,
    ) -> PhlowerTensor:
        """forward function which overload torch.nn.Module

        Args:
            data (IPhlowerTensorCollections):
                data which receives from predecessors
            supports (dict[str, PhlowerTensor]):
                sparse tensor objects

        Returns:
            PhlowerTensor:
                Tensor object
        """
        assert (
            len(data) <= 2
        ), f"At most two inputs are allowed. input: {len(data)}"

        supports = [field_data[name] for name in self._support_names]
        h = self._forward(data[0], supports=supports)

        if self._neumann_active:
            h = self._add_neumann(
                h,
                neumann_condition=data[1],
                inversed_moment=field_data[self._inversed_moment_name],
            )

        if not self._use_coefficient:
            return h

        h = self._forward_coefficient_network(h)
        return h

    def _forward(
        self, x: PhlowerTensor, supports: list[PhlowerTensor]
    ) -> PhlowerTensor:
        if self._mul_order == "ah_w":
            h = self._propagate(x, supports)
            _validate_rank0_before_applying_nonlinear(
                h,
                is_bias=self._weight.has_bias(),
                activations=self._weight.has_nonlinear_activations(),
            )
            return self._weight.forward(h)

        if self._mul_order == "a_hw":
            _validate_rank0_before_applying_nonlinear(
                x,
                is_bias=self._weight.has_bias(),
                activations=self._weight.has_nonlinear_activations(),
            )
            h = self._weight.forward(x)
            return self._propagate(h)

        raise NotImplementedError(
            f"multiplication order: {self._mul_order} is not implemeted."
        )

    def _forward_coefficient_network(self, x: PhlowerTensor) -> PhlowerTensor:
        if x.rank() == 0:
            coeff = self._coefficient_network.forward(x)
        else:
            # HACK Need to FIX ??
            coeff = self._coefficient_network.forward(self._contraction(x))

        return _functions.einsum(
            "i...f,if->i...f",
            x,
            coeff,
            dimension=x.dimension,
            is_time_series=x.is_time_series,
            is_voxel=x.is_voxel,
        )

    def _propagate(
        self, x: PhlowerTensor, supports: list[PhlowerTensor]
    ) -> PhlowerTensor:
        h = reduce(
            lambda y, f: f(y, supports),
            [self._select_propagations(name) for name in self._propagations],
            initial=x,
        )
        return h

    def _add_neumann(
        self,
        gradient: PhlowerTensor,
        neumann_condition: PhlowerTensor,
        inversed_moment: PhlowerTensor,
    ) -> PhlowerTensor:
        neumann_condition = torch.nan_to_num(neumann_condition, nan=0.0)
        # NOTE: Shape of inversed_moment is Shape(N, 3, 3, 1)
        neumann = (
            _functions.einsum(
                "ikl,il...f->ik...f",
                inversed_moment[..., 0],
                self._neumann_layer(neumann_condition),
                dimension=neumann_condition.dimension,
                is_time_series=neumann_condition.is_time_series,
                is_voxel=neumann_condition.is_voxel,
            )
            * self._neumann_factor
        )
        if self._neumann_apply_sigmoid_ratio:
            sigmoid_coeff = torch.sigmoid(self._coefficient_network[0].weight)
            return (
                sigmoid_coeff * gradient + (1.0 - sigmoid_coeff) * neumann
            ) * 2
        else:
            return gradient + neumann

    def _select_propagations(
        self, name: str | IsoGCNPropagationType
    ) -> Callable[[PhlowerTensor, list[PhlowerTensor]], PhlowerTensor]:
        if name == IsoGCNPropagationType.contraction:
            return self._contraction

        if name == IsoGCNPropagationType.convolution:
            return self._convolution

        if name == IsoGCNPropagationType.rotation:
            return self._rotation

        raise NotImplementedError(f"{name} is not implemented.")

    def _tensor_product(
        self, x: PhlowerTensor, supports: list[PhlowerTensor]
    ) -> PhlowerTensor:
        """Calculate tensor product G \\otimes x.

        Parameters
        ----------
        x: PhlowerTensor
            [n_vertex, dim, dim, ..., n_feature]-shaped tensor.
        supports: list[PhlowerTensor]
            List of [n_vertex, n_vertex]-shaped sparse tensor.

        Returns
        -------
        y: PhlowerTensor. The rank is incremeted by one from x.
        """
        shape = x.shape
        if x.rank() < 0:
            raise ValueError(f"Tensor shape invalid: {shape}")

        h = torch.stack(
            [_functions.spmm(support, x) for support in supports], dim=1
        )

        return h

    def _convolution(
        self, x: PhlowerTensor, supports: list[PhlowerTensor]
    ) -> PhlowerTensor:
        """Calculate convolution G \\ast x.

        Parameters
        ----------
        x: PhlowerTensor
            [n_vertex, n_feature]-shaped tensor.
        supports: list[torch.Tensor]
            List of [n_vertex, n_vertex]-shaped sparse tensor.

        Returns
        -------
        y: torch.Tensor
            [n_vertex, dim, n_feature]-shaped tensor.
        """
        return self._tensor_product(x, supports=supports)

    def _contraction(
        self, x: PhlowerTensor, supports: list[PhlowerTensor]
    ) -> PhlowerTensor:
        """Calculate contraction G \\cdot B. It calculates
        \\sum_l G_{i,j,k_1,k_2,...,l} H_{jk_1,k_2,...,l,f}

        Parameters
        ----------
        x: PhlowerTensor
            [n_vertex, dim, dim, ..., n_feature]-shaped tensor.
        supports: list[torch.Tensor]
            List of [n_vertex, n_vertex]-shaped sparse tensor.

        Returns
        -------
        y: PhlowerTensor
            [n_vertex, dim, ..., n_feature]-shaped tensor.
        """
        higher_h = self._tensor_product(x, supports=supports)
        return _functions.einsum("ikk...->i...", higher_h)

    def _rotation(
        self, x: PhlowerTensor, supports: list[PhlowerTensor]
    ) -> PhlowerTensor:
        """Calculate rotation G \\times x.

        Parameters
        ----------
        x: PhlowerTensor
            [n_vertex, dim, n_feature]-shaped tensor.
        supports: list[PhlowerTensor]
            List of [n_vertex, n_vertex]-shaped sparse tensor.

        Returns
        -------
        y: PhlowerTensor
            [n_vertex, dim, n_feature]-shaped tensor.
        """
        h = self._tensor_product(x, supports=supports)
        return torch.stack(
            [
                h[:, 1, 2] - h[:, 2, 1],
                h[:, 2, 0] - h[:, 0, 2],
                h[:, 0, 1] - h[:, 1, 0],
            ],
            dim=1,
        )


def _has_nonlinear_activations(activations: list[str]) -> bool:
    return len(v != _functions.identity.__name__ for v in activations) > 0


def _validate_rank0_before_applying_nonlinear(
    x: PhlowerTensor, is_bias: bool, activations: list[str]
) -> None:
    is_nonlinear_activations = _has_nonlinear_activations(activations)
    if x.rank() == 0:
        return

    if is_nonlinear_activations or is_bias:
        raise ValueError(
            "Cannot apply nonlinear operator for rank > 0 tensor."
            "Set bias and actications to "
            "apply linear operation for rank > 0 tensor"
        )
