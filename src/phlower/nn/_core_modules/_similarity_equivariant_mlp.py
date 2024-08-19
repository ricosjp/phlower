from __future__ import annotations

import numpy as np
import torch
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.nn._core_modules import _functions
from phlower.nn._core_modules._en_equivariant_mlp import EnEquivariantMLP
from phlower.nn._core_modules._identity import Identity
from phlower.nn._core_modules._mlp import MLP
from phlower.nn._core_modules._proportional import Proportional
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
        bias: bool = False,
        create_linear_weight: bool = False,
        norm_function_name: str = "identity",
        disable_en_equivariance: bool = False,
        invariant: bool = False,
        centering: bool = False,
    ) -> None:
        super().__init__()

        self._nodes = nodes
        self._disable_en_equivariance = disable_en_equivariance
        self._invariant = invariant
        self._centering = centering
        self._create_linear_weight = create_linear_weight

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
        supports: dict[str, PhlowerTensor] | None = None,
        dict_scales: dict[str, PhlowerTensor | float] | None = None,
        **kwards,
    ) -> PhlowerTensor:
        """forward function which overloads torch.nn.Module

        Args:
            data (IPhlowerTensorCollections):
                Data which receives from predecessors.
            supports (dict[str, PhlowerTensor], optional):
                Graph object. Defaults to None.

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

        dict_scales = {
            k: v ** dict_dimension[k] for k, v in dict_scales.items()
        }

        # Make h dimensionless
        for v in dict_scales.values():
            h = _functions.tensor_times_scalar(h, 1 / v)

        if self._centering:
            volume = dict_dimension["L"] ** 3
            mean = _functions.spatial_mean(
                _functions.tensor_times_scalar(h, volume)
            )
            h = h - mean

        h = self._mlp(phlower_tensor_collection({"h": h}))
        assert h.dimension.is_dimensionless()

        if self._centering:
            linear_mean = self._linear_weight(
                phlower_tensor_collection({"mean": mean})
            )
            h = h + linear_mean

        if self._invariant:
            return h

        # Come back to the original dimension
        for v in dict_scales.values():
            h = _functions.tensor_times_scalar(h, v)

        return h
