import pathlib
from collections.abc import Callable

import pytest
from hypothesis import given
from hypothesis import strategies as st
from phlower._base import PhysicalDimensions
from phlower.data import (
    DataLoaderBuilder,
    LazyPhlowerDataset,
    LumpedTensorData,
)
from phlower.settings import (
    ModelIOSetting,
    PhlowerPredictorSetting,
    PhlowerTrainerSetting,
)
from phlower.utils.enums import ModelSelectionType


@st.composite
def trainer_setting(draw: Callable) -> PhlowerTrainerSetting:
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
def predictor_setting(draw: Callable) -> PhlowerPredictorSetting:
    selection_type = draw(st.sampled_from(ModelSelectionType))
    target_epoch = (
        draw(st.integers(min_value=0))
        if selection_type == ModelSelectionType.SPECIFIED
        else None
    )
    setting = PhlowerPredictorSetting(
        selection_mode=selection_type.value,
        non_blocking=draw(st.booleans()),
        device=draw(st.text()),
        batch_size=draw(st.integers(min_value=1)),
        num_workers=draw(st.integers(min_value=1)),
        target_epoch=target_epoch,
    )
    return setting


@given(trainer_setting())
def test__create_from_trainer_setting(setting: PhlowerTrainerSetting):
    dataloader = DataLoaderBuilder.from_setting(setting)

    assert dataloader._non_blocking == setting.non_blocking
    assert dataloader._random_seed == setting.random_seed
    assert dataloader._batch_size == setting.batch_size
    assert dataloader._num_workers == setting.num_workers


@given(predictor_setting())
def test__create_from_predictor_setting(setting: PhlowerPredictorSetting):
    dataloader = DataLoaderBuilder.from_setting(setting)

    assert dataloader._non_blocking == setting.non_blocking
    assert dataloader._random_seed == setting.random_seed
    assert dataloader._batch_size == setting.batch_size
    assert dataloader._num_workers == setting.num_workers


def _to_modelIO_settings(
    names: list[tuple[str, int, dict]] | None,
) -> list[ModelIOSetting] | None:
    if names is None:
        return None
    return [
        ModelIOSetting(
            name=v,
            physical_dimension=dims,
            members=[{"name": v, "n_last_dim": n_dim}],
        )
        for v, n_dim, dims in names
    ]


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test__consider_batch_size(
    batch_size: int,
    create_tmp_dataset: None,
    output_base_directory: pathlib.Path,
):
    directories = [
        output_base_directory / v for v in ["data0", "data1", "data2"]
    ]
    dataset = LazyPhlowerDataset(
        input_settings=_to_modelIO_settings(
            [("x0", 1, {}), ("x1", 1, {}), ("x2", 1, {})]
        ),
        label_settings=_to_modelIO_settings([("y0", 1, {})]),
        directories=directories,
        field_settings=_to_modelIO_settings(
            [("s0", None, {}), ("s1", None, {})]
        ),
    )

    builder = DataLoaderBuilder(
        non_blocking=False,
        random_seed=0,
        batch_size=batch_size,
        num_workers=1,
    )
    dataloader = builder.create(dataset, drop_last=True)

    for item in dataloader:
        item: LumpedTensorData
        assert len(item.data_directories) == batch_size
        assert item.n_data == batch_size


@pytest.mark.parametrize(
    "x_variables, y_variables, fields, disable_dimensions, desired",
    [
        (
            [
                ("x0", 1, {"L": 2, "T": -2}),
                ("x1", 1, {"M": 2}),
                ("x2", 1, {"I": 1}),
            ],
            [
                ("y0", 1, {"N": -2}),
            ],
            [("s0", None, {"I": 1})],
            False,
            {
                "x0": PhysicalDimensions({"L": 2, "T": -2}),
                "x1": PhysicalDimensions({"M": 2}),
                "x2": PhysicalDimensions({"I": 1}),
                "y0": PhysicalDimensions({"N": -2}),
                "s0": PhysicalDimensions({"I": 1}),
            },
        )
    ],
)
def test__consider_dimensions(
    x_variables: list,
    y_variables: list,
    fields: list,
    disable_dimensions: bool,
    desired: dict,
    create_tmp_dataset: None,
    output_base_directory: pathlib.Path,
):
    directories = [
        output_base_directory / v for v in ["data0", "data1", "data2"]
    ]
    dataset = LazyPhlowerDataset(
        input_settings=_to_modelIO_settings(x_variables),
        label_settings=_to_modelIO_settings(y_variables),
        directories=directories,
        field_settings=_to_modelIO_settings(fields),
    )

    builder = DataLoaderBuilder(
        non_blocking=False,
        random_seed=0,
        batch_size=1,
        num_workers=1,
    )
    dataloader = builder.create(
        dataset,
        device="cpu",
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

        for data_name in item.field_data.keys():
            phydim = item.field_data[data_name].dimension.to_physics_dimension()
            assert phydim == desired[data_name]


@pytest.mark.parametrize(
    "inputs, labels, fields, disable_dimensions",
    [
        (
            [
                ("x0", 1, {"L": 2, "T": -2}),
                ("x1", 1, {"M": 2}),
                ("x2", 1, {"I": 1}),
            ],
            [("y0", 1, {"N": -2})],
            [("s0", None, {"I": 1})],
            True,
        )
    ],
)
def test__not_consider_dimensions(
    inputs: list,
    labels: list,
    fields: list,
    disable_dimensions: bool,
    create_tmp_dataset: None,
    output_base_directory: pathlib.Path,
):
    directories = [
        output_base_directory / v for v in ["data0", "data1", "data2"]
    ]
    dataset = LazyPhlowerDataset(
        input_settings=_to_modelIO_settings(inputs),
        label_settings=_to_modelIO_settings(labels),
        directories=directories,
        field_settings=_to_modelIO_settings(fields),
    )

    builder = DataLoaderBuilder(
        non_blocking=False,
        random_seed=0,
        batch_size=1,
        num_workers=1,
    )
    dataloader = builder.create(
        dataset,
        device="cpu",
        disable_dimensions=disable_dimensions,
    )

    for item in dataloader:
        item: LumpedTensorData
        for data_name in item.x_data.keys():
            assert not item.x_data[data_name].has_dimension

        for data_name in item.y_data.keys():
            assert not item.y_data[data_name].has_dimension

        for data_name in item.field_data.keys():
            assert not item.field_data[data_name].has_dimension
