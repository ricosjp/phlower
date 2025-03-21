from __future__ import annotations

import numpy as np
import torch
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.nn._core_modules._en_equivariant_mlp import EnEquivariantMLP
from phlower.nn._core_modules._identity import Identity
from phlower.nn._core_modules._mlp import MLP
from phlower.nn._core_modules._proportional import Proportional
from phlower.nn._functionals import _functions
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import SimilarityEquivariantMLPSetting
from phlower.utils.enums import PhysicalDimensionSymbolType
from phlower.utils.exceptions import (
    PhlowerDimensionRequiredError,
    PhlowerInvalidArgumentsError,
)


class SimilarityEquivariantMLP(IPhlowerCoreModule, torch.nn.Module):
    """
    Similarity-equivariant Multi Layer Perceptron as in
    https://proceedings.mlr.press/v235/horie24a.html.

    Parameters
    ----------
    nodes: list[int]
        List of feature dimension sizes (The last value of tensor shape).
    activations: list[str] | None (optional)
        List of activation functions to apply to the output.
        Defaults to None.
    dropouts: list[float] | None (optional)
        List of dropout rates to apply to the output.
        Defaults to None.
    bias: bool
        Whether to use bias.
    create_linear_weight: bool
        Whether to create a linear weight.
    norm_function_name: str
        Name of the normalization function to apply to the output.
        Defaults to "identity".
    disable_en_equivariance: bool
        Whether to disable en-equivariance.
    invariant: bool
        Whether to make the output invariant to the input.
    centering: bool
        Whether to center the output.
    coeff_amplify: float
        Coefficient to amplify the output.

    Examples
    --------
    >>> similarity_equivariant_mlp = SimilarityEquivariantMLP(
    ...     nodes=[10, 20, 30],
    ...     activations=["relu", "relu", "relu"],
    ...     dropouts=[0.1, 0.1, 0.1],
    ...     bias=True,
    ...     create_linear_weight=True,
    ...     norm_function_name="identity",
    ...     disable_en_equivariance=False,
    ...     invariant=False,
    ...     centering=False,
    ...     coeff_amplify=1.0,
    ... )
    >>> similarity_equivariant_mlp(data)
    """

    @classmethod
    def from_setting(cls, setting: SimilarityEquivariantMLPSetting) -> Self:
        """Generate model from setting object

        Args:
            setting (SimilarityEquivariantMLPSetting): setting object

        Returns:
            Self: SimilarityEquivariantMLP object
        """
        return cls(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "SimilarityEquivariantMLP"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        nodes: list[int],
        activations: list[str] | None = None,
        dropouts: list[float] | None = None,
        bias: bool = True,
        create_linear_weight: bool = False,
        norm_function_name: str = "identity",
        disable_en_equivariance: bool = False,
        invariant: bool = False,
        centering: bool = False,
        coeff_amplify: float = 1.0,
    ) -> None:
        super().__init__()

        self._nodes = nodes
        self._disable_en_equivariance = disable_en_equivariance
        self._invariant = invariant
        self._centering = centering
        self._create_linear_weight = create_linear_weight
        self._coeff_amplify = coeff_amplify

        if self._disable_en_equivariance:
            self._mlp = MLP(
                nodes=nodes,
                activations=activations,
                dropouts=dropouts,
                bias=bias,
            )
            self._linear_weight = self._init_linear_weight()
        else:
            self._mlp = EnEquivariantMLP(
                nodes=nodes,
                activations=activations,
                dropouts=dropouts,
                bias=bias,
                create_linear_weight=create_linear_weight,
                norm_function_name=norm_function_name,
            )
            self._linear_weight = self._mlp._linear_weight

    def _init_linear_weight(self) -> Identity | Proportional:
        if not self._create_linear_weight:
            if self._nodes[0] != self._nodes[-1]:
                raise ValueError(
                    "First and last nodes are different. "
                    "Set create_linear_weight True."
                )
            return Identity()

        return Proportional([self._nodes[0], self._nodes[-1]])

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
        dict_scales: dict[str, PhlowerTensor | float] | None = None,
        **kwards,
    ) -> PhlowerTensor:
        """forward function which overloads torch.nn.Module

        Args:
            data: IPhlowerTensorCollections
                Data which receives from predecessors.
            field_data: ISimulationField | None
                Constant information through training or prediction

        Returns:
            PhlowerTensor: Tensor object
        """
        h = data.unique_item()
        if not h.has_dimension:
            raise PhlowerDimensionRequiredError("Dimension is required")
        if dict_scales is None or len(data) < 1:
            raise PhlowerInvalidArgumentsError(
                f"Scale inputs are required. Given: {dict_scales}, {kwards}"
            )
        if not np.all(
            [
                PhysicalDimensionSymbolType.is_exist(k)
                for k in dict_scales.keys()
            ]
        ):
            raise PhlowerInvalidArgumentsError(
                "keys in dict_scales should be in "
                "PhysicalDimensionSymbolType. Given: {dict_scales.keys()}"
            )
        dict_dimension = h.dimension.to_dict()

        dict_scales_for_convert = {
            k: v ** dict_dimension[k] for k, v in dict_scales.items()
        }

        # Make h dimensionless
        for v in dict_scales_for_convert.values():
            h = _functions.tensor_times_scalar(h, 1 / v)

        if self._centering:
            volume = dict_dimension["L"] ** 3
            mean = _functions.spatial_mean(
                _functions.tensor_times_scalar(h, volume)
            )
            h = h - mean

        h = self._mlp(phlower_tensor_collection({"h": h * self._coeff_amplify}))
        assert h.dimension.is_dimensionless

        if self._centering:
            linear_mean = self._linear_weight(
                phlower_tensor_collection({"mean": mean})
            )
            h = h + linear_mean

        if self._invariant:
            return h

        # Come back to the original dimension
        for v in dict_scales_for_convert.values():
            h = _functions.tensor_times_scalar(h, v)

        return h
