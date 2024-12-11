import hypothesis.strategies as st
import numpy as np
import pytest
import torch
from hypothesis import given
from phlower import phlower_tensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import Rearrange, TimeSeriesToFeatures
from phlower.settings._module_settings import RearrangeSetting


@pytest.mark.parametrize(
    "pattern, axes_lengths",
    [
        (
            "time feature -> (feature time)",
            {"time": 10},
        ),
        (
            "time feature f2 f3 -> f2 f3 (feature time)",
            {"f2": 10, "f3": 20},
        ),
    ],
)
@given(
    n_last_feature=st.integers(min_value=1),
    save_as_time_series=st.booleans(),
    save_as_voxel=st.booleans(),
)
def test__can_pass_parameters_correctly(
    pattern: str,
    axes_lengths: dict[str, int],
    n_last_feature: int,
    save_as_time_series: bool,
    save_as_voxel: bool,
):
    setting = RearrangeSetting(
        pattern=pattern,
        n_last_feature=n_last_feature,
        save_as_time_series=save_as_time_series,
        save_as_voxel=save_as_voxel,
        axes_lengths=axes_lengths,
    )

    model = Rearrange.from_setting(setting)

    assert model._save_as_time_series == save_as_time_series
    assert model._save_as_voxel == save_as_voxel
    assert model._pattern == pattern
    assert model._axes_lengths == axes_lengths


def test__can_call_parameters():
    model = Rearrange(
        pattern="t v -> v t",
        axes_lengths={"t": 3},
        save_as_time_series=False,
        save_as_voxel=False,
    )

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "input_shape, pattern, axes_lengths, "
    "n_last_feature, save_as_time_series, save_as_voxel, "
    "desired_shape",
    [
        (
            (3, 15, 10),
            "time vertex feature -> vertex (time feature)",
            {"feature": 10},
            30,
            False,
            False,
            (15, 30),
        ),
        (
            (3, 15, 10, 5),
            "time vertex f1 f2 -> time vertex (f1 f2)",
            {"f1": 10},
            50,
            True,
            False,
            (3, 15, 50),
        ),
        (
            (3, 2, 1, 4, 10),
            "time x y z feature -> x y z (time feature)",
            {"feature": 10},
            30,
            False,
            True,
            (2, 1, 4, 30),
        ),
        (
            (3, 2, 1, 4, 4, 10),
            "time x y z f1 f2 -> time x y z (f1 f2)",
            {"f1": 4},
            40,
            True,
            True,
            (3, 2, 1, 4, 40),
        ),
    ],
)
def test__accessed(
    input_shape: tuple[int],
    pattern: str,
    axes_lengths: dict[str, int],
    n_last_feature: int,
    save_as_time_series: bool,
    save_as_voxel: bool,
    desired_shape: tuple[int],
):
    model = Rearrange(
        pattern=pattern,
        axes_lengths=axes_lengths,
        n_last_feature=n_last_feature,
        save_as_time_series=save_as_time_series,
        save_as_voxel=save_as_voxel,
    )

    phlower_tensors = phlower_tensor_collection(
        {
            "input": phlower_tensor(
                torch.from_numpy(np.random.rand(*input_shape))
            )
        }
    )
    output = model.forward(phlower_tensors)

    assert output.is_time_series == save_as_time_series
    assert output.is_voxel == save_as_voxel
    assert output.shape[-1] == n_last_feature
    assert output.shape == desired_shape


@pytest.mark.parametrize(
    "input_shape, pattern", [((3, 10, 2), "time vertex f -> vertex (f time)")]
)
def test__same_output_with_time_series_to_feature(
    input_shape: tuple[int], pattern: str
):
    model = Rearrange(
        pattern=pattern,
        axes_lengths={},
        save_as_time_series=False,
        save_as_voxel=False,
    )
    time2feature = TimeSeriesToFeatures()

    phlower_tensors = phlower_tensor_collection(
        {
            "input": phlower_tensor(
                torch.from_numpy(np.random.rand(*input_shape)),
                is_time_series=True,
            )
        }
    )
    output = model.forward(phlower_tensors)
    output_ts2f = time2feature.forward(phlower_tensors)

    np.testing.assert_array_almost_equal(output, output_ts2f)
