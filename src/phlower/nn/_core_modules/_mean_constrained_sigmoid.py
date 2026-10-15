from __future__ import annotations

import torch
from phlower_tensor import ISimulationField, PhlowerTensor, phlower_tensor
from phlower_tensor.collections import IPhlowerTensorCollections
from phlower_tensor.functionals import einsum, spatial_mean

from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import MeanConstrainedSigmoidSetting


class MeanConstrainedSigmoid(IPhlowerCoreModule, torch.nn.Module):
    """
    MeanConstrainedSigmoid projects the input feature to (0, 1)-valued
    features with its mean equaling to the given mean.
    It applies sigmoid(a * h + b), with b determined by solving the equation
    mean sigmoid(a * h + b) = target_mean.

    Parameters
    ----------
    output_feature_dim: int
        Output feature dimension.
    reference_variable_name: str
        Variable name to compute the target mean.
    weight_field_name: str | None = None
        Name of weight (e.g., volume) data in field_data.
    slope: float = 2.0
        Slope of the linear transformation before sigmoid. If learnable_slope
        is True, it is used as an initial value.
    learnable_slope: bool = False
        If True, lean slope.
    convergence_threshold: float = 1e-5
        Threshould for the Newton solver to obtain the target mean.
    max_iteration: int = 100
        Max iteration for the Newton solver.
    epsilon: float = 1.0e-6
        Small value to suppress numerical instability.

    Examples
    --------
    >>> mix = MeanConstrainedSigmoid(reference_variable_name="a")
    >>> mix(data)
    """

    @classmethod
    def from_setting(
        cls, setting: MeanConstrainedSigmoidSetting
    ) -> MeanConstrainedSigmoid:
        """Create MeanConstrainedSigmoid from setting object

        Args:
            setting (MeanConstrainedSigmoidSetting): setting object

        Returns:
            Self: MeanConstrainedSigmoid
        """
        return MeanConstrainedSigmoid(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of MeanConstrainedSigmoid

        Returns:
            str: name
        """
        return "MeanConstrainedSigmoid"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        output_feature_dim: int,
        reference_variable_name: str,
        weight_field_name: str | None = None,
        slope: float = 2.0,
        learnable_slope: bool = False,
        convergence_threshold: float = 1e-5,
        max_iteration: int = 100,
        epsilon: float = 1.0e-6,
        **kwards,
    ) -> None:
        super().__init__()
        self._output_feature_dim = output_feature_dim
        self._reference_variable_name = reference_variable_name
        self._weight_field_name = weight_field_name
        self._learnable_slope = learnable_slope

        self._convergence_threshold = convergence_threshold
        self._max_iteration = max_iteration
        self._epsilon = epsilon

        self._slope = phlower_tensor(self._initialize_slope(slope))

    def _initialize_slope(self, slope: float) -> torch.nn.Parameter:
        if self._learnable_slope:
            return torch.nn.Parameter(
                # * 2 to have almost the same mean as the input slope
                torch.rand(self._output_feature_dim) * slope * 2,
                requires_grad=True,
            )
        return torch.nn.Parameter(
            torch.tensor([slope] * self._output_feature_dim),
            requires_grad=False,
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
        reference_variable = data[self._reference_variable_name]
        variables = [
            v for k, v in data.items() if k != self._reference_variable_name
        ]
        if len(variables) != 1:
            raise ValueError(
                "Input variables should be 2 (1 reference + 1 variables).\n"
                f"(given: {data.items()}"
            )
        x = variables[0]

        if self._weight_field_name is None:
            weight = None
        else:
            weight = field_data[self._weight_field_name]

        reference_mean = self._compute_mean(reference_variable, weight)
        y = self._mean_constrained_sigmoid(x, reference_mean, weight=weight)
        return y

    def _compute_mean(
        self, x: PhlowerTensor, weight: float | None
    ) -> PhlowerTensor:
        m = spatial_mean(x, weight=weight)
        if x.is_time_series:
            m = torch.mean(m, axis=0)
        return m

    def _mean_constrained_sigmoid(
        self,
        logits: PhlowerTensor,
        reference_mean: PhlowerTensor,
        weight: PhlowerTensor | None = None,
    ) -> PhlowerTensor:
        if torch.any(reference_mean < 0.0):
            raise ValueError(
                f"Reference mean is negative so infeasible: {reference_mean}"
            )
        if torch.any(reference_mean > 1.0):
            raise ValueError(
                "Reference mean is greater than 1 so infeasible: "
                f"{reference_mean}"
            )

        # Center logits for numerical stability
        t = einsum(
            "f,...f->...f",
            torch.abs(self._slope.to(dtype=logits.dtype)),
            logits - spatial_mean(logits, weight=weight),
        )

        # Find b numerically, without storing the solver graph
        with torch.no_grad():
            q = torch.logit(reference_mean)
            lo = q - torch.max(t, dim=t.shape_pattern.nodes_dim)[0]
            hi = q - torch.min(t, dim=t.shape_pattern.nodes_dim)[0]

            for _ in range(self._max_iteration):
                mid = lo + 0.5 * (hi - lo)
                below = (
                    spatial_mean(torch.sigmoid(t + mid), weight=weight)
                    < reference_mean
                )

                lo = torch.where(below, mid, lo)
                hi = torch.where(below, hi, mid)

                if torch.all(
                    spatial_mean(torch.abs(hi - lo), weight=weight)
                    < self._convergence_threshold
                ):
                    break
            b0 = lo + 0.5 * (hi - lo)

        # Differentiable Newton correction
        p = torch.sigmoid(t + b0)

        derivative = torch.clamp(
            spatial_mean(p * torch.sigmoid(-t - b0), weight=weight),
            min=self._epsilon,
        )

        b = b0 + (reference_mean - spatial_mean(p, weight=weight)) / derivative

        y = torch.sigmoid(t + b)

        return y
