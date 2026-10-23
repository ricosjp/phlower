from __future__ import annotations

import abc
import pathlib
from typing import NamedTuple, TypeVar

import numpy as np
import torch
from phlower_tensor import PhlowerTensor, phlower_tensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.data import LumpedTensorData
from phlower.io import PhlowerDirectory
from phlower.services.preprocessing import ScalersComposition
from phlower.services.preprocessing._scalers import PhlowerScalerWrapper
from phlower.settings import ArrayDataIOSetting, NoiseSetting, PhlowerSetting
from phlower.utils.enums import PhlowerScalerName

_STD_SCALERS = (PhlowerScalerName.standardize, PhlowerScalerName.std_scale)
# NOTE: Scalers linear in the data, through which the noise can be scaled.
# isoam_scale is linear too, but scales sparse operators, not nodal values.
_LINEAR_SCALERS = (
    PhlowerScalerName.identity,
    PhlowerScalerName.max_abs_powered,
    PhlowerScalerName.min_max,
    PhlowerScalerName.standardize,
    PhlowerScalerName.std_scale,
    PhlowerScalerName.physical_nondimensionalize,
)

_Data = TypeVar("_Data", LumpedTensorData, IPhlowerTensorCollections)


class _NoiseItem(NamedTuple):
    setting: NoiseSetting
    input_scaler: PhlowerScalerWrapper
    increment_scaler: PhlowerScalerWrapper


class INoiseAwareScaler(metaclass=abc.ABCMeta):
    """Scaler adding noise to the inputs during training, and rescaling
    the data with the std of the noisy data. See NoiseSetting.
    """

    @abc.abstractmethod
    def apply(self, data: _Data, add_noise: bool = False) -> _Data:
        """Rescale the inputs, and the increment labels of a batch.
        A new noise is added to a batch when add_noise is True (training).
        """

    @abc.abstractmethod
    def remove_training_scaling(
        self, outputs: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        """Undo the rescaling of the predicted increments,
        before the inverse scaling."""

    @abc.abstractmethod
    def set_rank(self, rank: int) -> None:
        """Draw different noise in each process of parallel training."""


class NoneNoiseAwareScaler(INoiseAwareScaler):
    """Training without noise, where nothing is added or rescaled."""

    def apply(self, data: _Data, add_noise: bool = False) -> _Data:
        return data

    def remove_training_scaling(
        self, outputs: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        return outputs

    def set_rank(self, rank: int) -> None:
        return


class NoiseAwareScaler(INoiseAwareScaler):
    """Scaler for training.noise, built from the fitted scalers."""

    @classmethod
    def from_setting(cls, setting: PhlowerSetting) -> NoiseAwareScaler:
        if setting.scaling is None:
            raise ValueError(
                "Noise needs the scaling setting fitted at preprocessing."
            )

        scalers = ScalersComposition.from_setting(setting)
        items = []
        for noise in setting.training.noise:
            if noise.fraction is None:
                raise ValueError(
                    f"fraction of the noise on {noise.input_variable} "
                    "is not set. It is written by the trainer."
                )
            items.append(
                _NoiseItem(
                    setting=noise,
                    input_scaler=_get_scaler(
                        scalers,
                        setting,
                        _single_member_name(
                            setting.model.inputs, noise.input_variable
                        ),
                    ),
                    increment_scaler=_get_scaler(
                        scalers,
                        setting,
                        _single_member_name(
                            setting.model.labels, noise.increment_variable
                        ),
                    ),
                )
            )
        return cls(items, seed=setting.training.random_seed)

    def __init__(self, items: list[_NoiseItem], seed: int = 0) -> None:
        self._items = items
        self._seed = seed
        self._rank = 0
        # NOTE: An own generator makes the noise independent of
        # the other random numbers.
        self._generator = torch.Generator().manual_seed(seed)

    def set_rank(self, rank: int) -> None:
        if rank == self._rank:
            return
        self._rank = rank
        self._generator.manual_seed(self._seed + rank)

    def apply(self, data: _Data, add_noise: bool = False) -> _Data:
        if isinstance(data, LumpedTensorData):
            return self._apply_to_batch(data, add_noise)
        if add_noise:
            raise ValueError(
                "Noise is added only to a batch, which has the mask fields."
            )
        return self._rescale_inputs(data)

    def remove_training_scaling(
        self, outputs: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        for item in self._items:
            name = item.setting.increment_variable
            if name not in outputs:
                raise ValueError(f"Increment {name} is not an output.")
            _, r = self._scale_factors(item, is_input=False, data=outputs[name])
            outputs = outputs | phlower_tensor_collection(
                {name: outputs[name] / r}
            )
        return outputs

    def _apply_to_batch(
        self, batch: LumpedTensorData, add_noise: bool
    ) -> LumpedTensorData:
        x_data = dict(batch.x_data.items())
        y_data = dict(batch.y_data.items()) if batch.y_data else {}

        for item in self._items:
            noise = item.setting
            x = x_data[noise.input_variable]
            y = y_data.get(noise.increment_variable)
            physical_noise = (
                self._draw_noise(x, batch, noise) if add_noise else None
            )
            x_data[noise.input_variable] = self._apply_scaled_noise(
                physical_noise, item, is_input=True, data=x
            )
            if y is not None:
                y_data[noise.increment_variable] = self._apply_scaled_noise(
                    physical_noise, item, is_input=False, data=y
                )

        return LumpedTensorData(
            x_data=phlower_tensor_collection(x_data),
            field_data=batch.field_data,
            data_directories=batch.data_directories,
            y_data=(
                phlower_tensor_collection(y_data)
                if batch.y_data is not None
                else None
            ),
            x_batch_info=batch.x_batch_info,
            y_batch_info=batch.y_batch_info,
        )

    def _rescale_inputs(
        self, inputs: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        for item in self._items:
            name = item.setting.input_variable
            _, r = self._scale_factors(item, is_input=True, data=inputs[name])
            inputs = inputs | phlower_tensor_collection(
                {name: inputs[name] * r}
            )
        return inputs

    def _apply_scaled_noise(
        self,
        physical_noise: torch.Tensor | None,
        item: _NoiseItem,
        is_input: bool,
        data: PhlowerTensor,
    ) -> PhlowerTensor:
        """r (data + noise / s) for the input,
        r (data - noise / s) for the increment, and
        r data without noise."""
        inv_s, r = self._scale_factors(item, is_input=is_input, data=data)
        if physical_noise is not None:
            scaled_noise = _like(physical_noise * inv_s, data)
            data = data + scaled_noise if is_input else data - scaled_noise
        return data * r

    def _scale_factors(
        self, item: _NoiseItem, is_input: bool, data: PhlowerTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """1 / s and r = s / s' of the input or the increment of the item.
        r is 1 for scalers other than standardize and std_scale."""
        noise = item.setting
        scaler = item.input_scaler if is_input else item.increment_scaler
        nodes_dim = data.shape_pattern.nodes_dim
        tail_shape = tuple(data.shape[nodes_dim + 1 :])
        # NOTE: transform(x) is (x - c) / s for a linear scaler,
        # so transform(1) - transform(0) = 1 / s.
        # The scalers are only used through their transform, so that
        # the factors have the layout of the scaled data.
        ones = np.ones((1, *tail_shape))
        inv_s = (
            scaler.transform(ones).to_numpy()
            - scaler.transform(np.zeros_like(ones)).to_numpy()
        )
        # NOTE: Only the std scalers hold the std of the data,
        # from which the std of the noisy data is computed.
        if scaler.method_name in _STD_SCALERS:
            r = 1.0 / np.sqrt(1.0 + noise.fraction * (noise.std * inv_s) ** 2)
        else:
            r = np.ones_like(inv_s)
        return (
            torch.as_tensor(inv_s, device=data.device, dtype=data.dtype),
            torch.as_tensor(r, device=data.device, dtype=data.dtype),
        )

    def _draw_noise(
        self, data: PhlowerTensor, batch: LumpedTensorData, noise: NoiseSetting
    ) -> torch.Tensor:
        """Gaussian noise in the unit of the unscaled data,
        zero outside the mask."""
        drawn = noise.std * torch.randn(
            data.shape, generator=self._generator, dtype=data.dtype
        ).to(device=data.device)
        if noise.mask is None:
            return drawn

        if noise.mask not in batch.field_data:
            raise ValueError(f"Mask field {noise.mask} is not found.")
        return drawn * _node_mask(batch.field_data[noise.mask], data)


def create_noise_aware_scaler(setting: PhlowerSetting) -> INoiseAwareScaler:
    """NoiseAwareScaler of training.noise, or NoneNoiseAwareScaler
    without noise."""
    if setting.training is None or len(setting.training.noise) == 0:
        return NoneNoiseAwareScaler()
    return NoiseAwareScaler.from_setting(setting)


def compute_noise_fractions(
    noise_settings: list[NoiseSetting],
    directories: list[pathlib.Path],
    decrypt_key: bytes | None = None,
) -> list[float]:
    """Fraction of the entries with noise over the directories,
    for each noise setting."""
    fractions = []
    for noise in noise_settings:
        if noise.mask is None:
            fractions.append(1.0)
            continue

        n_masked = n_total = 0
        for directory in directories:
            mask = (
                PhlowerDirectory(directory)
                .find_variable_file(noise.mask)
                .load(decrypt_key=decrypt_key)
                .to_numpy()
            )
            if mask.ndim != 2 or mask.shape[1] != 1:
                raise ValueError(
                    f"Mask {noise.mask} must have the shape (n_nodes, 1), "
                    f"got {mask.shape} in {directory}."
                )
            if ((mask != 0) & (mask != 1)).any():
                raise ValueError(
                    f"Mask {noise.mask} must be 0 or 1 in {directory}. "
                    "It must not be scaled."
                )
            n_masked += int(np.count_nonzero(mask))
            n_total += mask.size
        if n_total == 0:
            raise ValueError(f"Mask {noise.mask} is empty.")
        fractions.append(n_masked / n_total)
    return fractions


def _single_member_name(
    io_settings: list[ArrayDataIOSetting], name: str
) -> str:
    settings = [v for v in io_settings if v.name == name]
    if len(settings) != 1:
        raise ValueError(f"Variable {name} of the noise is not found.")
    setting = settings[0]
    if len(setting.members) != 1:
        raise ValueError(f"Variable {name} of the noise must have one member.")
    if setting.random_sampling.is_active:
        raise ValueError(
            f"Noise on {name} cannot be used with random sampling."
        )
    return setting.members[0].name


def _get_scaler(
    scalers: ScalersComposition, setting: PhlowerSetting, member_name: str
) -> PhlowerScalerWrapper:
    scaler_name = setting.scaling.get_scaler_name(member_name)
    if scaler_name is None:
        raise ValueError(f"Scaler of {member_name} is not found.")

    scaler = scalers.get_scaler(scaler_name)
    if scaler.method_name not in _LINEAR_SCALERS:
        raise ValueError(
            f"Scaler of {member_name} must be one of "
            f"{', '.join(_LINEAR_SCALERS)} for the noise, "
            f"not {scaler.method_name}."
        )
    if not scaler.is_fitted():
        raise ValueError(
            f"Scaler of {member_name} is not fitted. Set the scaling "
            "setting fitted at preprocessing (preprocess.yml)."
        )
    return scaler


def _like(tensor: torch.Tensor, reference: PhlowerTensor) -> PhlowerTensor:
    """Wrap a tensor with the dimension and the flags of the reference,
    so that it can be added to it."""
    return phlower_tensor(
        tensor,
        dimension=reference.dimension,
        is_time_series=reference.is_time_series,
        is_voxel=reference.is_voxel,
    )


def _node_mask(mask: PhlowerTensor, data: PhlowerTensor) -> torch.Tensor:
    nodes_dim = data.shape_pattern.nodes_dim
    if mask.shape[-1] != 1 or mask.shape[0] != data.shape[nodes_dim]:
        raise ValueError(
            f"Mask of shape {tuple(mask.shape)} does not match "
            f"the data of shape {tuple(data.shape)}."
        )

    shape = [1] * data.ndim
    shape[nodes_dim] = -1
    return (mask.to_tensor().flatten() != 0).to(data.dtype).reshape(shape)
