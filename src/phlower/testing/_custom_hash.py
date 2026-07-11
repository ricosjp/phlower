import hashlib
from typing import Any

import torch

# noqa: ANN401


def assert_hash_equal(actual: Any, desired: Any):  # noqa: ANN401
    """
    Assert that the hash of the actual and desired values are equal.
    This function computes the SHA-256 hash of the provided
    actual and desired values.
    Many types are supported, including torch.Tensor, dict, list, tuple, set,
    int, float, str, bool, and None.
    Thus, this is useful for comparing state_dicts of models, optimizers,
    and other objects in PyTorch.
    """

    actual_hash = _get_hash(actual)
    desired_hash = _get_hash(desired)

    assert actual_hash == desired_hash, (
        f"Hash mismatch: {actual_hash} != {desired_hash}. "
        f"Actual: {actual}, Desired: {desired}"
    )


def _get_hash(inputs: Any) -> str:  # noqa: ANN401
    h = hashlib.sha256()
    _update_hash(h, inputs)
    return h.hexdigest()


def _update_hash(h: "hashlib.Hash", value: Any):  # noqa: ANN401
    match value:
        case torch.Tensor():
            value = value.detach().cpu().contiguous()
            _update_tensor_hash(h, value)
        case dict():
            for key in sorted(value.keys()):
                h.update(str(key).encode())
                _update_hash(h, value[key])
        case list() | tuple() | set():
            for v in value:
                _update_hash(h, v)
        case int() | float() | str() | bool():
            h.update(str(value).encode())
        case None:
            h.update(b"None")
        case _:
            raise TypeError(f"Unsupported type {type(value)} in state_dict.")


def _update_tensor_hash(h: "hashlib.Hash", tensor: torch.Tensor):
    h.update(str(tensor.dtype).encode())
    h.update(str(tuple(tensor.shape)).encode())
    h.update(tensor.numpy().tobytes())
