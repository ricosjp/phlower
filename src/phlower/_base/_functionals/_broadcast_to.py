from phlower._base import IPhlowerTensor
from phlower._base.tensors import PhlowerShapePattern


def broadcast_to(
    x: IPhlowerTensor, shape_pattern: PhlowerShapePattern
) -> IPhlowerTensor:
    """Broadcast a tensor to a given shape
     only applied for feature dimensions.

    Args:
        x (IPhlowerTensor): The input tensor.
        shape (PhlowerShapePattern): The target shape.

    Returns:
        IPhlowerTensor: The broadcasted tensor.
    """

    if x.is_voxel != shape_pattern.is_voxel:
        raise ValueError(
            f"Cannot broadcast tensor with voxel shape "
            f"{x.shape_pattern} to target shape {shape_pattern}"
        )

    feature_shape = x.shape[x.shape_pattern.feature_start_dim :]
    target_shape = shape_pattern.shape[shape_pattern.feature_start_dim :]

    if len(target_shape) < len(feature_shape):
        raise ValueError(
            f"Cannot broadcast tensor with feature shape "
            f"{feature_shape} to target shape {target_shape}"
        )

    n_fill = len(target_shape) - len(feature_shape)
    new_shape = (
        x.shape[: x.shape_pattern.feature_start_dim]
        + (1,) * n_fill
        + feature_shape
    )
    return x.reshape(
        new_shape,
        is_time_series=x.is_time_series,
        is_voxel=x.is_voxel,
    )
