import numpy as np
import pytest
from phlower_tensor import phlower_array

from phlower.data._sampler import RandomPointSampler
from phlower.settings import ArrayDataIOSetting


@pytest.mark.parametrize(
    "n_sampled_points, is_time_series, shape, expected_shape",
    [
        (10, False, (100, 10), (10, 10)),
        (20, True, (10, 80, 20, 1), (10, 20, 20, 1)),
        (30, True, (5, 50, 3), (5, 30, 3)),
    ],
)
def test__random_sampler_has_different_index(
    n_sampled_points: int,
    is_time_series: bool,
    shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
):
    io_setting = ArrayDataIOSetting(
        name="feature0",
        members=[{"name": "feature0"}],
        is_time_series=is_time_series,
        is_voxel=False,
        random_sampling={
            "is_active": True,
            "n_sampled_points": n_sampled_points,
        },
    )
    arr = phlower_array(
        data=np.random.rand(*shape),
        is_time_series=is_time_series,
        is_voxel=False,
        dimensions=None,
    )

    sampler = RandomPointSampler()
    result = sampler.sample(io_setting=io_setting, array=arr)
    assert result.shape == expected_shape

    sampler = RandomPointSampler()
    result2 = sampler.sample(io_setting=io_setting, array=arr)
    assert result2.shape == expected_shape

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(result.to_numpy(), result2.to_numpy())


@pytest.mark.parametrize(
    "n_sampled_points, is_time_series, shape, expected_shape",
    [
        (10, False, (100, 10), (10, 10)),
        (20, True, (10, 80, 20, 1), (10, 20, 20, 1)),
        (30, True, (5, 50, 3), (5, 30, 3)),
    ],
)
def test__random_sampler_with_same_as(
    n_sampled_points: int,
    is_time_series: bool,
    shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
):

    arr = phlower_array(
        data=np.random.rand(*shape),
        is_time_series=is_time_series,
        is_voxel=False,
        dimensions=None,
    )
    io_setting = ArrayDataIOSetting(
        name="feature0",
        members=[{"name": "feature0"}],
        is_time_series=is_time_series,
        is_voxel=False,
        random_sampling={
            "n_sampled_points": n_sampled_points,
        },
    )
    sampler = RandomPointSampler()
    result = sampler.sample(io_setting=io_setting, array=arr)

    io_setting2 = ArrayDataIOSetting(
        name="feature1",
        members=[{"name": "feature1"}],
        is_time_series=is_time_series,
        is_voxel=False,
        random_sampling={
            "same_as": "feature0",
        },
    )
    result2 = sampler.sample(io_setting=io_setting2, array=arr)

    assert result.shape == result2.shape
    assert result.shape == expected_shape
    np.testing.assert_array_equal(result.to_numpy(), result2.to_numpy())


@pytest.mark.parametrize("n_sampled_points", [10, 20, 100])
def test__random_sampler_has_unique_index(
    n_sampled_points: int,
):

    arr = phlower_array(
        data=np.random.rand(100, 10),
        is_time_series=False,
        is_voxel=False,
        dimensions=None,
    )
    io_setting = ArrayDataIOSetting(
        name="feature0",
        members=[{"name": "feature0"}],
        is_time_series=False,
        is_voxel=False,
        random_sampling={
            "is_active": True,
            "n_sampled_points": n_sampled_points,
        },
    )

    sampler = RandomPointSampler()
    _ = sampler.sample(io_setting=io_setting, array=arr)

    assert (
        np.unique(sampler._history["feature0"].sampled_indices).size
        == n_sampled_points
    )


def test__sample_with_same_node():
    features = phlower_array(
        data=np.random.rand(100, 10),
        is_time_series=False,
        is_voxel=False,
        dimensions=None,
    )
    io_setting0 = ArrayDataIOSetting(
        name="feature0",
        members=[{"name": "feature0"}],
        is_time_series=False,
        is_voxel=False,
        random_sampling={
            "is_active": True,
            "n_sampled_points": 10,
        },
    )
    t_features = phlower_array(
        data=np.random.rand(10, 100, 10),
        is_time_series=True,
        is_voxel=False,
        dimensions=None,
    )
    io_setting1 = ArrayDataIOSetting(
        name="t_feature1",
        members=[{"name": "t_feature1"}],
        is_time_series=True,
        is_voxel=False,
        random_sampling={
            "same_as": "feature0",
        },
    )

    sampler = RandomPointSampler()
    _ = sampler.sample(io_setting=io_setting0, array=features)
    result1 = sampler.sample(io_setting=io_setting1, array=t_features)

    assert result1.to_numpy().shape[1] == 10


def test__sample_with_reverse_order():
    features = phlower_array(
        data=np.random.rand(100, 10),
        is_time_series=False,
        is_voxel=False,
        dimensions=None,
    )
    io_setting0 = ArrayDataIOSetting(
        name="feature0",
        members=[{"name": "feature0"}],
        is_time_series=False,
        is_voxel=False,
        random_sampling={
            "is_active": True,
            "n_sampled_points": 10,
        },
    )
    t_features = phlower_array(
        data=np.random.rand(10, 100, 10),
        is_time_series=True,
        is_voxel=False,
        dimensions=None,
    )
    io_setting1 = ArrayDataIOSetting(
        name="t_feature1",
        members=[{"name": "t_feature1"}],
        is_time_series=True,
        is_voxel=False,
        random_sampling={
            "same_as": "feature0",
        },
    )
    sampler = RandomPointSampler()
    # sample with t_feature1
    result1 = sampler.sample(io_setting=io_setting1, array=t_features)
    assert result1.to_numpy().shape[1] == 10

    result0 = sampler.sample(io_setting=io_setting0, array=features)
    assert result0.to_numpy().shape[0] == 10


def test__not_allowed_voxel_data():
    features = phlower_array(
        data=np.random.rand(10, 10, 10, 3),
        is_time_series=False,
        is_voxel=True,
        dimensions=None,
    )
    io_setting0 = ArrayDataIOSetting(
        name="feature0",
        members=[{"name": "feature0"}],
        is_time_series=False,
        is_voxel=True,
        random_sampling={
            "is_active": True,
            "n_sampled_points": 10,
        },
    )
    sampler = RandomPointSampler()
    with pytest.raises(
        ValueError, match="Random sampling is not supported for voxel data"
    ):
        _ = sampler.sample(io_setting=io_setting0, array=features)


@pytest.mark.parametrize(
    "feature0_shape, is_time_series0, feature1_shape, is_time_series1",
    [
        ((100, 10), False, (10, 200, 10), True),
        ((10, 200, 10), True, (100, 10), False),
        ((10, 200, 10), True, (10, 100, 10), True),
    ],
)
def test__raise_error_when_inconsistent_n_points(
    feature0_shape: tuple[int, ...],
    is_time_series0: bool,
    feature1_shape: tuple[int, ...],
    is_time_series1: bool,
):
    features = phlower_array(
        data=np.random.rand(*feature0_shape),
        is_time_series=is_time_series0,
        dimensions=None,
    )
    io_setting0 = ArrayDataIOSetting(
        name="feature0",
        members=[{"name": "feature0"}],
        is_time_series=False,
        is_voxel=False,
        random_sampling={
            "is_active": True,
            "n_sampled_points": 10,
        },
    )
    io_setting1 = ArrayDataIOSetting(
        name="t_feature1",
        members=[{"name": "t_feature1"}],
        is_time_series=True,
        is_voxel=False,
        random_sampling={
            "same_as": "feature0",
        },
    )
    sampler = RandomPointSampler()
    _ = sampler.sample(io_setting=io_setting0, array=features)

    features1 = phlower_array(
        data=np.random.rand(*feature1_shape),
        is_time_series=is_time_series1,
    )
    with pytest.raises(ValueError, match="Number of points does not match"):
        _ = sampler.sample(io_setting=io_setting1, array=features1)
