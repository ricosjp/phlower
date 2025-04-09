from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

from phlower._base.tensors import PhlowerTensor, phlower_tensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _utils
from phlower.nn._functionals import _functions
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings._isogcn_setting import (
    IsoGCNPropagationType,
    IsoGCNSetting,
)


class IsoGCN(IPhlowerCoreModule, torch.nn.Module):
    """IsoGCN is a neural network module that performs
    an spatial pifferential operation on the input tensor.
    Ref: https://arxiv.org/abs/2005.06316

    Parameters
    ----------
    nodes: list[int]
        List of feature dimension sizes (The last value of tensor shape).
    isoam_names: list[str]
        List of names of the isoam tensors.
    propagations (list[IsoGCNPropagationType]):
        List of propagations to apply to the input tensor.
    use_self_network: bool
        Whether to use the self network.
    self_network_activations: list[str] | None (optional)
        List of activation functions to apply to the output of the self-network.
        Defaults to None.
    self_network_dropouts: list[float] | None (optional)
        List of dropout rates to apply to the output of the self-network.
        Defaults to None.
    self_network_bias: bool | None (optional)
        Whether to use bias in the self-network.
        Defaults to False.
    use_coefficient: bool
        Whether to use the coefficient network.
    coefficient_activations: list[str] | None (optional)
        List of activation functions to apply to the output of
        the coefficient network.
        Defaults to None.
    coefficient_dropouts: list[float] | None (optional)
        List of dropout rates to apply to the output of the coefficient network.
        Defaults to None.
    coefficient_bias: bool | None (optional)
        Whether to use bias in the coefficient network.
        Defaults to False.
    mul_order (Literal["ah_w", "a_hw"]):
        Multiplication order.
        Defaults to "ah_w".
    to_symmetric: bool
        Whether to make the output symmetric.
        Defaults to False.
    use_neumann: bool
        Whether to use the Neumann layer.
        Defaults to False.
    neumann_factor: float
        Factor to multiply the Neumann layer.
        Defaults to 1.0.
    neumann_input_name: str | None (optional)
        Name of the input tensor for the Neumann layer.
        Defaults to None.
    inversed_moment_name: str
        Name of the inversed moment tensor.
        Defaults to "".

    Examples
    --------
    >>> # Create an IsoGCN instance for laplacian operation with neumann condition
    >>> isogcn = IsoGCN(
    ...     nodes=[10, 20, 30],                    # Feature dimensions
    ...     isoam_names=["isoam_x", "isoam_y", "isoam_z"],
    ...     propagations=[                          # Types of propagation
    ...         IsoGCNPropagationType.convolution,
    ...         IsoGCNPropagationType.contraction
    ...     ],
    ...     # Self network configuration
    ...     use_self_network=True,
    ...     self_network_activations=["relu", "relu"],
    ...     self_network_dropouts=[0.1, 0.1],
    ...     self_network_bias=True,
    ...     # Coefficient network configuration
    ...     use_coefficient=True,
    ...     coefficient_activations=["relu", "relu"],
    ...     coefficient_dropouts=[0.1, 0.1],
    ...     coefficient_bias=True,
    ...     # Additional settings
    ...     mul_order="ah_w",
    ...     to_symmetric=True,
    ...     use_neumann=True,
    ...     neumann_factor=1.0,
    ...     neumann_input_name="neumann_input",
    ...     inversed_moment_name="inversed_moment"
    ... )
    >>> # Forward pass
    >>> output = isogcn(data)
    """  # noqa

    @classmethod
    def from_setting(cls, setting: IsoGCNSetting) -> IsoGCN:
        """Create IsoGCN from IsoGCNSetting instance

        Args:
            setting (GCNSetting): setting object for IsoGCN

        Returns:
            IsoGCN: IsoGCN object
        """
        return IsoGCN(
            nodes=setting.nodes,
            isoam_names=setting.isoam_names,
            propagations=setting.propagations,
            use_self_network=setting.self_network.use_network,
            self_network_activations=setting.self_network.activations,
            self_network_dropouts=setting.self_network.dropouts,
            self_network_bias=setting.self_network.bias,
            use_coefficient=setting.coefficient_network.use_network,
            coefficient_activations=setting.coefficient_network.activations,
            coefficient_dropouts=setting.coefficient_network.dropouts,
            coefficient_bias=setting.coefficient_network.bias,
            mul_order=setting.mul_order,
            to_symmetric=setting.to_symmetric,
            use_neumann=setting.neumann_setting.use_neumann,
            neumann_factor=setting.neumann_setting.factor,
            neumann_input_name=setting.neumann_setting.neumann_input_name,
            inversed_moment_name=setting.neumann_setting.inversed_moment_name,
        )

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
        nodes: list[int] | None,
        isoam_names: list[str],
        propagations: list[IsoGCNPropagationType],
        use_self_network: bool,
        self_network_activations: list[str] | None = None,
        self_network_dropouts: list[str] | None = None,
        self_network_bias: bool | False = False,
        use_coefficient: bool = False,
        coefficient_activations: list[str] | None = None,
        coefficient_dropouts: list[str] | None = None,
        coefficient_bias: bool | None = False,
        mul_order: Literal["ah_w", "a_hw"] = "ah_w",
        to_symmetric: bool = False,
        use_neumann: bool = False,
        neumann_factor: float = 1.0,
        neumann_input_name: str | None = None,
        inversed_moment_name: str = "",
    ) -> None:
        super().__init__()

        self._use_self_network = use_self_network
        if self._use_self_network:
            self._self_network = _utils.ExtendedLinearList(
                nodes=[nodes[0], nodes[-1]],
                activations=self_network_activations,
                dropouts=self_network_dropouts,
                bias=self_network_bias,
            )
        else:
            self._self_network = None

        self._use_coefficient = use_coefficient
        if self._use_coefficient:
            self._coefficient_network = _utils.ExtendedLinearList(
                nodes=nodes,
                activations=coefficient_activations,
                dropouts=coefficient_dropouts,
                bias=coefficient_bias,
            )
        else:
            self._coefficient_network = None

        self._propagations = propagations
        self._nodes = nodes
        self._isoam_names = isoam_names
        self._mul_order = mul_order
        self._to_symmetric = to_symmetric

        self._inversed_moment_name = inversed_moment_name
        self._use_neumann = use_neumann
        self._neumann_factor = neumann_factor
        self._neumann_input_name = neumann_input_name
        if self._use_neumann:
            self._neumann_layer = self._create_neumann_layer()
        else:
            self._neumann_layer = None

    def _create_neumann_layer(
        self,
    ) -> Callable[[PhlowerTensor], PhlowerTensor] | None:
        if self._self_network is None:
            raise ValueError(
                "Use self_network when neumannn layer is necessary. "
                "It is because neumann layer refers to weight of self_network."
            )
        return self._self_network[0]

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None: ...

    def get_reference_name(self) -> str | None:
        return None

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None,
        **kwards,
    ) -> PhlowerTensor:
        """forward function which overload torch.nn.Module

        Args:
            data: IPhlowerTensorCollections
                data which receives from predecessors
            field_data: ISimulationField | None
                Constant information through training or prediction

        Returns:
            PhlowerTensor:
                Tensor object
        """
        assert (
            len(data) <= 2
        ), f"At most two inputs are allowed. input: {len(data)}"

        supports = [field_data[name] for name in self._isoam_names]
        neumann_value = data.pop(self._neumann_input_name, None)
        x = data.unique_item()

        h = self._forward_self_network(x, supports=supports)

        if self._use_neumann:
            h = self._add_neumann(
                h,
                neumann_tensor=neumann_value,
                inversed_moment=field_data[self._inversed_moment_name],
            )

        if self._to_symmetric:
            # it may be used for velocity
            h = (h + h.rearrange("n x1 x2 f -> n x2 x1 f")) / 2.0

        if not self._use_coefficient:
            return h

        coeff = self._forward_coefficient_network(x)
        return _functions.einsum(
            "i...f,if->i...f",
            h,
            coeff,
            dimension=h.dimension,
            is_time_series=h.is_time_series,
            is_voxel=h.is_voxel,
        )

    def _forward_self_network(
        self, x: PhlowerTensor, supports: list[PhlowerTensor]
    ) -> PhlowerTensor:
        if not self._use_self_network:
            return self._propagate(x, supports)

        assert self._self_network is not None

        if self._mul_order == "ah_w":
            h = self._propagate(x, supports)
            _validate_rank0_before_applying_nonlinear(
                h, layer=self._self_network
            )
            return self._self_network.forward(h)

        if self._mul_order == "a_hw":
            _validate_rank0_before_applying_nonlinear(
                x, layer=self._self_network
            )
            h = self._self_network.forward(x)
            return self._propagate(h)

        raise NotImplementedError(
            f"multiplication order: {self._mul_order} is not implemeted."
        )

    def _forward_coefficient_network(self, x: PhlowerTensor) -> PhlowerTensor:
        if x.rank() == 0:
            coeff = self._coefficient_network.forward(x)
        else:
            coeff = self._coefficient_network.forward(_functions.contraction(x))

        return coeff

    def _propagate(
        self, x: PhlowerTensor, supports: list[PhlowerTensor]
    ) -> PhlowerTensor:
        h = x
        for name in self._propagations:
            h = self._select_propagations(name)(h, supports)
        return h

    def _add_neumann(
        self,
        gradient: PhlowerTensor,
        neumann_tensor: PhlowerTensor,
        inversed_moment: PhlowerTensor,
    ) -> PhlowerTensor:
        neumann_tensor = torch.nan_to_num(neumann_tensor, nan=0.0)
        # NOTE: Shape of inversed_moment is Shape(N, 3, 3)
        neumann = (
            _functions.einsum(
                "ikl,il...f->ik...f",
                inversed_moment,
                self._neumann_layer(neumann_tensor),
                dimension=neumann_tensor.dimension,
                is_time_series=neumann_tensor.is_time_series,
                is_voxel=neumann_tensor.is_voxel,
            )
            * self._neumann_factor
        )
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

        if name == IsoGCNPropagationType.tensor_product:
            return self._tensor_product

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
        return phlower_tensor(
            torch.stack(
                [
                    h[:, 1, 2] - h[:, 2, 1],
                    h[:, 2, 0] - h[:, 0, 2],
                    h[:, 0, 1] - h[:, 1, 0],
                ],
                dim=1,
            ),
            dimension=h.dimension,
            is_time_series=h.is_time_series,
            is_voxel=h.is_voxel,
        )


def _validate_rank0_before_applying_nonlinear(
    x: PhlowerTensor, layer: _utils.ExtendedLinearList
) -> None:
    if x.rank() == 0:
        return

    if layer.has_nonlinearity():
        raise ValueError(
            "Nonlinear operation is not allowed for rank > 0 tensor. "
            "Try to apply linear operation for rank > 0 tensor. "
            f"Layer info: {layer.has_bias()=}, {layer.has_dropout()=}, "
            f"activations={layer.get_activation_names()}"
        )

    return
