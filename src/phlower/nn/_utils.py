import pathlib
from collections.abc import Callable
from typing import TypeVar

import numpy as np
from phlower_tensor.collections import IPhlowerTensorCollections

from phlower.nn._interface_module import IPhlowerModuleAdapter
from phlower.settings._debug_parameter_setting import (
    PhlowerModuleDebugParameters,
)
from phlower.utils import get_logger
from phlower.utils.calculation_state import CalculationState
from phlower.utils.exceptions import PhlowerRunTimeError
from phlower.utils.typing import PhlowerForwardHook

_logger = get_logger(__name__)


T1 = TypeVar("T1")
T2 = TypeVar("T2")


def attach_location_to_error_message(
    func: Callable[[IPhlowerModuleAdapter, T1], T2],
) -> Callable[[IPhlowerModuleAdapter, T1], T2]:
    """Decorator to modify the error message of a function.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The decorated function with modified error message.
    """

    def wrapper(self: IPhlowerModuleAdapter, *args: T1, **kwargs) -> T2:
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if isinstance(e, PhlowerRunTimeError):
                e.add_location(self.name)
                raise e
            else:
                ex = PhlowerRunTimeError()
                ex.set_message(str(e))
                ex.add_location(self.name)
                raise ex from e

    return wrapper


# region Debug Helper


def attach_debug_helper(
    ref: IPhlowerModuleAdapter, params: PhlowerModuleDebugParameters
) -> None:
    if params.output_tensor_shape:
        ref.register_phlower_forward_hook(
            _CheckOutputTensorShapeHook(params.output_tensor_shape)
        )

    _register_hook = None
    if params.dump_forward_tensor:
        _register_hook = _DumpEnRouteTensorHook()
        ref.register_phlower_forward_hook(_register_hook.forward_save())

    if params.dump_backward_tensor:
        # HACK: This implementation is not ideal.
        # However, register_full_backward_hook is not available
        # because PhlowerModuleAdapter does not return pure tensor.
        # Following warning is raised when register_full_backward_hook is called
        #
        # UserWarning: For backward hooks to be called, module output should be
        # a Tensor or a tuple of Tensors but received
        # <class 'phlower_tensor.collections.tensors.\
        # _tensor_collections.PhlowerDictTensors'>
        #
        # Similar issues are repoted in huggingface transformers.
        # https://github.com/huggingface/transformers/issues/36508

        _register_hook = _register_hook or _DumpEnRouteTensorHook()
        ref.register_phlower_forward_hook(
            _register_hook.forward_hook_for_backward()
        )
        ref.register_phlower_finalize_debug_hook(_register_hook.backward_hook())


class _CheckOutputTensorShapeHook(PhlowerForwardHook):
    def __init__(self, output_tensor_shape: list[int]):
        self._output_tensor_shape = output_tensor_shape

    def __call__(
        self,
        name: str,
        unique_name: str,
        result: IPhlowerTensorCollections,
        *,
        state: CalculationState | None = None,
    ) -> None:

        assert len(result) == 1, (
            "Only one output tensor is supported for shape check"
        )
        _result = result.unique_item()

        if len(self._output_tensor_shape) != len(_result.shape):
            raise ValueError(
                f"In {name}, result tensor shape "
                f"{_result.shape} is different from desired shape "
                f"{self._output_tensor_shape} "
                "in yaml file"
            )

        output_tensor_shape = np.array(self._output_tensor_shape)
        positive_flag = output_tensor_shape > 0
        output_tensor_shape = np.where(
            positive_flag, output_tensor_shape, _result.shape
        )
        if not all(output_tensor_shape == _result.shape):
            ind = np.where(output_tensor_shape != _result.shape)[0][0]

            raise ValueError(
                f"In {name}, {ind}-th result tensor shape "
                f"{_result.shape} is different from desired shape "
                f"{output_tensor_shape} in yaml file"
            )


class _DumpEnRouteTensorHook(PhlowerForwardHook):
    def __init__(self):
        self._forward_debug_base_path: pathlib.Path | None = None
        self._ref_result: list[IPhlowerTensorCollections] = []
        self._prev_batch_idx: int = -1

    def forward_save(
        self,
    ) -> PhlowerForwardHook:

        def _hook(
            name: str,
            unique_name: str,
            result: IPhlowerTensorCollections,
            *,
            state: CalculationState | None = None,
        ):
            if state is None:
                _logger.info(
                    "Dumping en route tensor is skipped"
                    " because calculation state is not provided."
                )
                return

            self._forward_debug_base_path = _determine_base_output_directory(
                state, unique_name
            )
            output_directory = _increment_if_duplicated(
                self._forward_debug_base_path
            )

            for output_name, ph_tensor in result.items():
                output_path = output_directory / f"{output_name}.npy"
                np.save(output_path, ph_tensor.numpy())

        return _hook

    def forward_hook_for_backward(
        self,
    ) -> PhlowerForwardHook:
        def _hook(
            name: str,
            unique_name: str,
            result: IPhlowerTensorCollections,
            *,
            state: CalculationState | None = None,
        ):
            if state is None or state.mode != "training":
                return

            if self._forward_debug_base_path is None:
                self._forward_debug_base_path = (
                    _determine_base_output_directory(state, unique_name)
                )

            if self._prev_batch_idx != state.current_batch_iteration:
                self._ref_result = []
                self._prev_batch_idx = state.current_batch_iteration

            for v in result.values():
                v._tensor.retain_grad()
            self._ref_result.append(result)

        return _hook

    def backward_hook(
        self,
    ) -> Callable[[], None]:

        def _hook():
            if self._forward_debug_base_path is None:
                raise ValueError(
                    "Dumping grad tensor is unavailable because forward hook "
                    "is not called yet."
                )

            for idx, ref_result in enumerate(self._ref_result):
                output_directory = self._forward_debug_base_path / f"{idx}"
                output_directory.mkdir(parents=True, exist_ok=True)

                for k, v in ref_result.items():
                    output_path = output_directory / f"{k}_grad_output.npy"
                    np.save(output_path, v._tensor.grad.numpy())

            return None

        return _hook


def _determine_base_output_directory(
    state: CalculationState, unique_name: str
) -> pathlib.Path:
    output_directory = (
        state.output_directory
        / f"en_route_tensors/epoch={state.current_epoch}/"
        f"batch_idx={state.current_batch_iteration}/"
        f"{unique_name}"
    )
    return output_directory


def _increment_if_duplicated(output_directory: pathlib.Path) -> pathlib.Path:
    if not output_directory.exists():
        output_directory = output_directory / "0"
        output_directory.mkdir(parents=True)
        return output_directory

    nums = [
        int(p.name)
        for p in output_directory.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]

    latest = 0 if len(nums) == 0 else sorted(nums)[-1] + 1

    output_directory = output_directory / str(latest)
    output_directory.mkdir(parents=True)
    return output_directory


# endregion
