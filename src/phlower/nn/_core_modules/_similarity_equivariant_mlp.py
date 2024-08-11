from __future__ import annotations

import numpy as np
import torch
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.nn._core_modules import (
    MLP,
    EnEquivariantMLP,
    Identity,
    Proportional,
    _functions,
)
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
        norm_function_name: str = None,
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
                nodes=nodes, activations=activations,
                dropouts=dropouts, bias=bias,
            )
            self._linear_weight = self._init_linear_weight()
        else:
            self._mlp = EnEquivariantMLP(
                nodes=nodes, activations=activations,
                dropouts=dropouts, bias=bias,
                create_linear_weight=create_linear_weight,
                norm_function_name=norm_function_name,
            )
            self._linear_weight = self._mlp._linear_weight

    def _init_linear_weight(self):
        if not self._create_linear_weight:
            if self._nodes[0] != self._nodes[-1]:
                raise ValueError(
                    "First and last nodes are different. "
                    "Set create_linear_weight True.")
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
        **kwards,
    ) -> PhlowerTensor:
        """forward function which overloads torch.nn.Module

        Args:
            data (IPhlowerTensorCollections):
                Data with the keys in tensor or PhysicalDimensionSymbolType,
                with tensor is the input tensor to be computed.
                PhysicalDimensionSymbolType keys denote physical scales.
                No need to input all dimensions, but at least one scale
                information should be input.
            supports (dict[str, PhlowerTensor], optional):
                Graph object. Defaults to None.

        Returns:
            PhlowerTensor: Tensor object
        """
        # TODO: Check if we can assume key convention
        h = data.pop("tensor")
        if not h.has_dimension:
            raise PhlowerDimensionRequiredError("Dimension is required")

        if len(data) < 1:
            raise PhlowerInvalidArgumentsError("Scale inputs are required")
        if not np.all([
                PhysicalDimensionSymbolType.is_exist(k) for k in data.keys()]):
            raise PhlowerInvalidArgumentsError(
                "keys should be in PhysicalDimensionSymbolType. "
                f"Given: {data.keys()}")
        dict_dimension = h.dimension.to_dict()

        dict_scales = {
            k: v**dict_dimension[k] for k, v in data.items()}

        # Make h dimensionless
        for v in dict_scales.values():
            h = _functions.tensor_times_scalar(h, 1 / v)

        if self._centering:
            volume = dict_dimension["L"]**3
            mean = _functions.spatial_mean(
                _functions.tensor_times_scalar(h, volume))
            h = h - mean

        h = self._mlp(phlower_tensor_collection({"h": h}))
        assert h.dimension.is_dimensionless()

        if self._centering:
            linear_mean = self._linear_weight(
                phlower_tensor_collection({"mean": mean}))
            h = h + linear_mean

        if self._invariant:
            return h

        # Come back to the original dimension
        for v in dict_scales.values():
            h = _functions.tensor_times_scalar(h, v)

        return h
