from __future__ import annotations

import torch
from typing_extensions import Self

from phlower._base._functionals import unbatch
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
    scale_names: dict[str, str]
        Name of scales to be used to convert input features to dimensionless.
        Keys should be PhysicalDimensionSymbolType (T, L, M, ...) and values
        should be names of fields.
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
    cross_interaction: bool
        If True, use interaction layer proposed by
        https://arxiv.org/abs/2203.06442.
    normalize: bool
        If True, use eq (12) in https://arxiv.org/abs/2203.06442 when
        cross_interaction is True.
    unbatch: bool
        If True, perform unbatching. Set False when you call this module
        from another module which perform unbaching. Note that this option
        cannot be set from yaml.
        Default: True

    Examples
    --------
    >>> similarity_equivariant_mlp = SimilarityEquivariantMLP(
    ...     nodes=[10, 20, 30],
    ...     scale_names={"T": "time", "L": "length"},
    ...     activations=["relu", "relu", "relu"],
    ...     dropouts=[0.1, 0.1, 0.1],
    ...     bias=True,
    ...     create_linear_weight=True,
    ...     norm_function_name="identity",
    ...     disable_en_equivariance=False,
    ...     invariant=False,
    ...     centering=False,
    ...     coeff_amplify=1.0,
    ...     cross_interaction=False,
    ...     normalize=True,
    ... )
    >>> similarity_equivariant_mlp(data, field_data=field_data)
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
        scale_names: dict[str, str],
        activations: list[str] | None = None,
        dropouts: list[float] | None = None,
        bias: bool = True,
        create_linear_weight: bool = False,
        norm_function_name: str = "identity",
        disable_en_equivariance: bool = False,
        invariant: bool = False,
        centering: bool = False,
        coeff_amplify: float = 1.0,
        cross_interaction: bool = False,
        normalize: bool = True,
        unbatch: bool = True,
    ) -> None:
        super().__init__()

        self._nodes = nodes
        self._scale_names = scale_names
        self._disable_en_equivariance = disable_en_equivariance
        self._invariant = invariant
        self._centering = centering
        self._create_linear_weight = create_linear_weight
        self._coeff_amplify = coeff_amplify
        self._unbatch = unbatch

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
                cross_interaction=cross_interaction,
                normalize=normalize,
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
        if field_data is None:
            raise PhlowerInvalidArgumentsError("Field data is required")

        h = data.unique_item()

        if not h.has_dimension:
            raise PhlowerDimensionRequiredError("Dimension is required")

        if self._unbatch:
            # NOTE: Assume length scale is always fed
            n_nodes = field_data.get_batched_n_nodes(self._scale_names["L"])
            unbatched_h = unbatch(h, n_nodes=n_nodes)
            unbatched_field = unbatch(field_data, n_nodes=n_nodes)
            return torch.cat(
                [
                    self._forward(_h, field_data=_f)
                    for _h, _f in zip(unbatched_h, unbatched_field, strict=True)
                ],
                dim=0,
            )
        return self._forward(h, field_data=field_data)

    def _forward(
        self, h: PhlowerTensor, field_data: ISimulationField
    ) -> PhlowerTensor:
        dict_scales = {
            scale_name: field_data[field_name]
            for scale_name, field_name in self._scale_names.items()
        }
        if len(dict_scales) == 0:
            raise PhlowerInvalidArgumentsError(
                f"Scale inputs are required in fields. Given: {field_data}"
            )
        dict_dimension = h.dimension.to_dict()

        dict_scales_for_convert = {
            k: v ** dict_dimension[k] for k, v in dict_scales.items()
        }

        # Make h dimensionless
        for v in dict_scales_for_convert.values():
            h = _functions.tensor_times_scalar(h, 1 / v)

        if self._centering:
            volume = dict_scales["L"] ** 3
            mean = _functions.spatial_mean(h, volume)
            h = h - mean

        h = self._mlp(phlower_tensor_collection({"h": h * self._coeff_amplify}))
        assert (
            h.dimension.is_dimensionless
        ), f"Feature cannot be converted to dimensionless: {h.dimension}"

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
