from __future__ import annotations

from typing import NamedTuple

import numpy as np
from phlower_tensor import IPhlowerArray, PhlowerTensor, phlower_tensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
    reduce_stack,
)

from phlower.data import LumpedTensorData
from phlower.nn import PhlowerGroupModule
from phlower.services.preprocessing import PhlowerScalingService
from phlower.settings import PhlowerModelSetting, RolloutSetting
from phlower.utils.calculation_state import CalculationState


class Rollout:
    """Prediction applying the model step by step."""

    @classmethod
    def from_setting(
        cls,
        setting: RolloutSetting,
        model_setting: PhlowerModelSetting,
        output_key_map: dict[str, str],
        scalers: PhlowerScalingService,
    ) -> Rollout:
        variables = _resolve_rollout_variables(
            setting,
            model_setting=model_setting,
            output_key_map=output_key_map,
            scalers=scalers,
        )
        return cls(variables, n_steps=setting.n_steps, scalers=scalers)

    def __init__(
        self,
        variables: list[_RolloutVariable],
        n_steps: int | None,
        scalers: PhlowerScalingService,
    ):
        self._variables = variables
        self._n_steps = n_steps
        self._scalers = scalers

    def state_key_map(self) -> dict[str, str]:
        return {v.state: v.state_scaler for v in self._variables}

    def run(
        self,
        model: PhlowerGroupModule,
        batch: LumpedTensorData,
        state: CalculationState,
    ) -> IPhlowerTensorCollections:
        variables = self._variables
        state_names = [v.state for v in variables]

        initial_states = batch.x_data.mask(state_names)
        currents = initial_states.snapshot(0)
        n_steps = _determine_n_steps(
            self._n_steps, initial_states.get_time_series_length()
        )

        other_inputs = phlower_tensor_collection(
            {k: v for k, v in batch.x_data.items() if k not in state_names}
        )
        for name, value in other_inputs.items():
            if value.is_time_series and value.time_series_length < n_steps:
                raise ValueError(
                    f"Time series input {name} is shorter than "
                    f"{n_steps} steps of rollout."
                )

        # NOTE: Keep the states in the unit of the data, so that scaling
        # back and forth does not accumulate errors.
        physical_states = _to_numpy_dict(
            self._scalers.inverse_transform(
                {v.state_scaler: currents[v.state].numpy() for v in variables}
            )
        )

        predicted: list[IPhlowerTensorCollections] = []
        for step in range(n_steps):
            inputs = other_inputs.snapshot(step)
            inputs.update(currents)
            h = model.forward(inputs, field_data=batch.field_data, state=state)

            increments = _to_numpy_dict(
                self._scalers.inverse_transform(
                    {
                        v.increment_scaler: h[v.increment].numpy()
                        for v in variables
                    }
                )
            )
            for v in variables:
                physical_state = physical_states[v.state_scaler]
                increment = increments[v.increment_scaler]
                if increment.shape != physical_state.shape:
                    raise ValueError(
                        f"Shape of the increment {v.increment} "
                        f"{increment.shape} does not match the state "
                        f"{v.state} {physical_state.shape}."
                    )
                physical_states[v.state_scaler] = physical_state + increment
            # NOTE: max_process=1 avoids a process pool at every step.
            scaled_states = self._scalers.transform(
                physical_states, max_process=1
            )
            currents = phlower_tensor_collection(
                {
                    v.state: _to_tensor_like(
                        scaled_states[v.state_scaler].to_numpy(),
                        currents[v.state],
                    )
                    for v in variables
                }
            )

            h.update(currents)
            # NOTE: Keep the outputs of the steps on cpu,
            # since a long rollout does not fit in the device memory.
            predicted.append(h.to("cpu"))

        model.finalize_debug()
        return reduce_stack(predicted, to_time_series=True)


class _RolloutVariable(NamedTuple):
    state: str
    """Name of the input updated at every step."""
    state_scaler: str
    """Name of the variable whose scaler applies to the state."""
    increment: str
    """Name of the output added to the state."""
    increment_scaler: str
    """Name of the variable whose scaler applies to the increment."""


def _resolve_rollout_variables(
    setting: RolloutSetting,
    model_setting: PhlowerModelSetting,
    output_key_map: dict[str, str],
    scalers: PhlowerScalingService,
) -> list[_RolloutVariable]:
    """Resolve the scalers of the states and the increments.

    output_key_map maps the outputs to the names of their scalers.
    """
    time_series_length = model_setting.network.time_series_length
    if time_series_length:
        raise ValueError(
            "Rollout needs a one-step model, but the network iterates "
            f"over time with time_series_length: {time_series_length}."
        )

    output_keys = model_setting.network.get_output_keys()
    # NOTE: The states are added to the prediction key map,
    # so their scalers must not be used by the outputs.
    used_scalers = set(output_key_map.values())
    variables = []
    for variable in setting.variables:
        state = variable.state_variable
        increment = variable.increment_variable
        inputs = [s for s in model_setting.inputs if s.name == state]
        if len(inputs) != 1 or len(inputs[0].members) != 1:
            raise ValueError(
                f"State of rollout: {state} must be an input "
                "with only one member."
            )
        if state in output_keys:
            raise ValueError(
                f"State of rollout: {state} must not be an output."
            )
        if increment not in output_keys:
            raise ValueError(
                f"Increment of rollout: {increment} is not an output."
            )
        if increment not in output_key_map:
            raise ValueError(
                f"Scaler of the increment: {increment} is not found. "
                "Set it in output_to_scaler_name."
            )

        state_scaler = inputs[0].members[0].name
        increment_scaler = output_key_map[increment]
        if state_scaler in used_scalers:
            raise ValueError(
                f"Scaler {state_scaler} of the state {state} "
                "is already used by an output or another state."
            )
        used_scalers.add(state_scaler)
        for name in (state_scaler, increment_scaler):
            if not scalers.is_scaler_exist(name):
                raise ValueError(
                    f"Scaler for {name} is not found in the scaling setting."
                )

        variables.append(
            _RolloutVariable(
                state=state,
                state_scaler=state_scaler,
                increment=increment,
                increment_scaler=increment_scaler,
            )
        )
    return variables


def _determine_n_steps(n_steps: int | None, time_length: int) -> int:
    """Number of steps, which defaults to the time length of the states."""
    if n_steps is not None:
        return n_steps
    if time_length == 0:
        raise ValueError(
            "Set n_steps of rollout when the states are not time series."
        )
    return time_length


def _to_numpy_dict(arrays: dict[str, IPhlowerArray]) -> dict[str, np.ndarray]:
    return {k: v.to_numpy() for k, v in arrays.items()}


def _to_tensor_like(
    array: np.ndarray, reference: PhlowerTensor
) -> PhlowerTensor:
    return phlower_tensor(
        array,
        dimension=reference.dimension,
        is_time_series=reference.is_time_series,
        is_voxel=reference.is_voxel,
        dtype=reference.dtype,
        device=reference.device,
    )
