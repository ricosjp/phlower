import pytest
from hypothesis import given
from hypothesis import strategies as st

from phlower._base import PhysicalDimensions
from phlower.data import (
    DataLoaderBuilder,
    LazyPhlowerDataset,
    LumpedTensorData,
)
from phlower.settings import PhlowerPredictorSetting, PhlowerTrainerSetting
from phlower.utils.enums import ModelSelectionType


@st.composite
def trainer_setting(draw):
    setting = PhlowerTrainerSetting(
        loss_setting={"name2loss": {"u": "mse"}},
        non_blocking=draw(st.booleans()),
        device=draw(st.text()),
        random_seed=draw(st.integers(min_value=0)),
        batch_size=draw(st.integers(min_value=1)),
        num_workers=draw(st.integers(min_value=1)),
    )
    return setting


@st.composite
def predictor_setting(draw):
    setting = PhlowerPredictorSetting(
        selection_mode=draw(st.sampled_from(ModelSelectionType)),
        non_blocking=draw(st.booleans()),
        device=draw(st.text()),
        batch_size=draw(st.integers(min_value=1)),
        num_workers=draw(st.integers(min_value=1)),
    )
    return setting


@given(trainer_setting())
def test__create_from_trainer_setting(setting: PhlowerTrainerSetting):
    dataloader = DataLoaderBuilder.from_setting(setting)

    assert dataloader._non_blocking == setting.non_blocking
    assert dataloader._device == setting.device
    assert dataloader._random_seed == setting.random_seed
    assert dataloader._batch_size == setting.batch_size
    assert dataloader._num_workers == setting.num_workers


@given(predictor_setting())
def test__create_from_predictor_setting(setting: PhlowerPredictorSetting):
    dataloader = DataLoaderBuilder.from_setting(setting)

    assert dataloader._non_blocking == setting.non_blocking
    assert dataloader._device == setting.device
    assert dataloader._random_seed == setting.random_seed
    assert dataloader._batch_size == setting.batch_size
    assert dataloader._num_workers == setting.num_workers


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test__consider_batch_size(
    batch_size, create_tmp_dataset, output_base_directory
):
    directories = [
        output_base_directory / v for v in ["data0", "data1", "data2"]
    ]
    dataset = LazyPhlowerDataset(
        x_variable_names=["x0", "x1", "x2"],
        y_variable_names=["y0"],
        directories=directories,
        support_names=["s0", "s1"],
    )

    builder = DataLoaderBuilder(
        non_blocking=False,
        device="cpu",
        random_seed=0,
        batch_size=batch_size,
        num_workers=1,
    )
    dataloader = builder.create(dataset, drop_last=True)

    for item in dataloader:
        item: LumpedTensorData
        assert len(item.data_directories) == batch_size


@pytest.mark.parametrize(
    "dimensions, disable_dimensions, desired",
    [
        (
            {
                "x0": {"length": 2, "time": -2},
                "x1": {"mass": 2},
                "x2": {"electric_current": 1},
                "y0": {"amount_of_substance": -2},
                "s0": {"electric_current": 1},
            },
            False,
            {
                "x0": PhysicalDimensions({"length": 2, "time": -2}),
                "x1": PhysicalDimensions({"mass": 2}),
                "x2": PhysicalDimensions({"electric_current": 1}),
                "y0": PhysicalDimensions({"amount_of_substance": -2}),
                "s0": PhysicalDimensions({"electric_current": 1}),
            },
        )
    ],
)
def test__consider_dimensions(
    dimensions,
    disable_dimensions,
    desired,
    create_tmp_dataset,
    output_base_directory,
):
    directories = [
        output_base_directory / v for v in ["data0", "data1", "data2"]
    ]
    dataset = LazyPhlowerDataset(
        x_variable_names=["x0", "x1", "x2"],
        y_variable_names=["y0"],
        directories=directories,
        support_names=["s0"],
    )

    builder = DataLoaderBuilder(
        non_blocking=False,
        device="cpu",
        random_seed=0,
        batch_size=1,
        num_workers=1,
    )
    dataloader = builder.create(
        dataset,
        variable_dimensions=dimensions,
        disable_dimensions=disable_dimensions,
    )

    for item in dataloader:
        item: LumpedTensorData
        for data_name in item.x_data.keys():
            phydim = item.x_data[data_name].dimension.to_physics_dimension()
            assert phydim == desired[data_name]

        for data_name in item.y_data.keys():
            phydim = item.y_data[data_name].dimension.to_physics_dimension()
            assert phydim == desired[data_name]

        for data_name in item.sparse_supports.keys():
            phydim = item.sparse_supports[
                data_name
            ].dimension.to_physics_dimension()
            assert phydim == desired[data_name]


@pytest.mark.parametrize(
    "dimensions, disable_dimensions",
    [
        (
            {
                "x0": {"length": 2, "time": -2},
                "x1": {"mass": 2},
                "x2": {"electric_current": 1},
                "y0": {"amount_of_substance": -2},
                "s0": {"electric_current": 1},
            },
            True,
        )
    ],
)
def test__not_consider_dimensions(
    dimensions, disable_dimensions, create_tmp_dataset, output_base_directory
):
    directories = [
        output_base_directory / v for v in ["data0", "data1", "data2"]
    ]
    dataset = LazyPhlowerDataset(
        x_variable_names=["x0", "x1", "x2"],
        y_variable_names=["y0"],
        directories=directories,
        support_names=["s0"],
    )

    builder = DataLoaderBuilder(
        non_blocking=False,
        device="cpu",
        random_seed=0,
        batch_size=1,
        num_workers=1,
    )
    dataloader = builder.create(
        dataset,
        variable_dimensions=dimensions,
        disable_dimensions=disable_dimensions,
    )

    for item in dataloader:
        item: LumpedTensorData
        for data_name in item.x_data.keys():
            assert not item.x_data[data_name].has_dimension

        for data_name in item.y_data.keys():
            assert not item.y_data[data_name].has_dimension

        for data_name in item.sparse_supports.keys():
            assert not item.sparse_supports[data_name].has_dimension
