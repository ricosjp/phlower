import pathlib
import sys
from collections.abc import Callable

import pytest
import torch
import yaml
from hypothesis import given
from hypothesis import strategies as st

from phlower.data import (
    DataLoaderBuilder,
    LazyPhlowerDataset,
    LumpedTensorData,
    OnMemoryPhlowerDataSet,
)
from phlower.settings import (
    ArrayDataIOSetting,
    MeshDataIOSetting,
    PhlowerPredictorSetting,
    PhlowerSetting,
    PhlowerTrainerSetting,
)
from phlower.utils._dimensions_class import PhysicalDimensions
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
) -> list[ArrayDataIOSetting] | None:
    if names is None:
        return None
    return [
        ArrayDataIOSetting(
            name=v,
            physical_dimension=dims,
            members=[{"name": v, "n_last_dim": n_dim}],
        )
        for v, n_dim, dims in names
    ]


def _to_meshIO_settings(
    names: list[str] | None,
) -> list[MeshDataIOSetting] | None:
    if names is None:
        return None
    return [
        MeshDataIOSetting(
            name=v,
            filename=f"{v}.vtk",
        )
        for v in names
    ]


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test__consider_batch_size(
    batch_size: int,
    create_base_dataset: None,
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
        num_workers=0,
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
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test__consider_dimensions(
    x_variables: list,
    y_variables: list,
    fields: list,
    disable_dimensions: bool,
    desired: dict,
    device: str,
    create_base_dataset: None,
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
        num_workers=0,
    )
    dataloader = builder.create(
        dataset,
        device=device,
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
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test__not_consider_dimensions(
    inputs: list,
    labels: list,
    fields: list,
    disable_dimensions: bool,
    device: str,
    create_base_dataset: None,
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
        num_workers=0,
    )
    dataloader = builder.create(
        dataset,
        device=device,
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


# region test for dataset with mesh data


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Not supported in Python versions below 3.12",
)
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test__load_dataset_with_mesh(
    device: str,
    create_dataset_with_mesh: None,
    output_base_directory_w_mesh: pathlib.Path,
):
    directories = [
        output_base_directory_w_mesh / v for v in ["data0", "data1", "data2"]
    ]
    dataset = LazyPhlowerDataset(
        input_settings=_to_modelIO_settings(
            [
                ("x0", 1, {"L": 2, "T": -2}),
                ("x1", 1, {"M": 2}),
                ("x2", 1, {"I": 1}),
            ]
        ),
        label_settings=_to_modelIO_settings([("y0", 1, {"N": -2})]),
        directories=directories,
        field_settings=_to_meshIO_settings(["mesh"]),
    )

    # num_workers must be 0 to pass coverage
    builder = DataLoaderBuilder(
        non_blocking=False,
        random_seed=0,
        batch_size=1,
        num_workers=0,
    )
    dataloader = builder.create(dataset, device=device)

    assert len(dataloader) > 0
    for item in dataloader:
        item: LumpedTensorData
        mesh = item.field_data.get_mesh()
        assert mesh is not None
        assert mesh.n_points > 0


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Not supported in Python versions below 3.12",
)
def test__raise_error_when_batch_size_larger_than_one_and_mesh_data_exists(
    create_dataset_with_mesh: None,
    output_base_directory_w_mesh: pathlib.Path,
):
    directories = [
        output_base_directory_w_mesh / v for v in ["data0", "data1", "data2"]
    ]
    dataset = LazyPhlowerDataset(
        input_settings=_to_modelIO_settings(
            [
                ("x0", 1, {"L": 2, "T": -2}),
                ("x1", 1, {"M": 2}),
                ("x2", 1, {"I": 1}),
            ]
        ),
        label_settings=_to_modelIO_settings([("y0", 1, {"N": -2})]),
        directories=directories,
        field_settings=_to_meshIO_settings(["mesh"]),
    )

    builder = DataLoaderBuilder(
        non_blocking=False,
        random_seed=0,
        batch_size=2,
        num_workers=0,
    )
    dataloader = builder.create(dataset, device="cpu")

    assert len(dataloader) > 0
    with pytest.raises(
        NotImplementedError,
        match="Batch size must be 1 when field data contains mesh data",
    ):
        _ = dataloader.__iter__().__next__()


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Not supported in Python versions below 3.12",
)
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
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test__disable_dimensions_for_mesh_data(
    inputs: list,
    labels: list,
    fields: list,
    disable_dimensions: bool,
    device: str,
    create_dataset_with_mesh: None,
    output_base_directory_w_mesh: pathlib.Path,
):
    directories = [
        output_base_directory_w_mesh / v for v in ["data0", "data1", "data2"]
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
        num_workers=0,
    )
    dataloader = builder.create(
        dataset,
        device=device,
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


# endregion


# region test for batch mode


_DATA_DIR = pathlib.Path(__file__).parent / "data"


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize(
    "yaml_file_name", ["sample_batch_mode_with_index_like.yml"]
)
def test__consider_batch_mode(
    batch_size: int,
    yaml_file_name: str,
    create_dataset_with_index_like_values: None,
    output_base_directory_index_like: pathlib.Path,
):
    directories = [
        output_base_directory_index_like / v
        for v in ["data0", "data1", "data2"]
    ]

    yaml_path = _DATA_DIR / yaml_file_name
    setting = PhlowerSetting.read_yaml(yaml_path)
    setting = PhlowerSetting.model_validate(
        setting.model_dump()
        | {
            "training": {
                "batch_size": batch_size,
            }
        }
    )

    dataset = OnMemoryPhlowerDataSet.create(
        input_settings=setting.model.inputs,
        label_settings=setting.model.labels,
        field_settings=setting.model.fields,
        directories=directories,
    )
    builder = DataLoaderBuilder.from_setting(setting.training)

    dataloader = builder.create(dataset, drop_last=True)

    index_like_names = [
        "index_like0",
        "index_like1",
    ]
    for item in dataloader:
        item: LumpedTensorData
        assert len(item.data_directories) == batch_size
        assert item.n_data == batch_size

        for name in index_like_names:
            assert item.field_data[name].dtype == torch.int32

            # check if the index_like values are shifted correctly
            val = torch.unique(item.field_data[name])
            _shape = item.field_data[name].shape
            assert val.shape == _shape
            assert torch.min(item.field_data[name]) == 0
            assert torch.max(item.field_data[name]) == _shape[0] - 1


# endregion


# region test for random sampling


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("yaml_file_name", ["sample_random_sampling.yml"])
def test__random_sampling(
    batch_size: int,
    yaml_file_name: str,
    create_dataset_with_n_nodes_100: pathlib.Path,
):
    output_base_directory = create_dataset_with_n_nodes_100
    directories = [
        output_base_directory / v for v in ["data0", "data1", "data2"]
    ]

    yaml_path = _DATA_DIR / yaml_file_name
    with yaml_path.open() as f:
        yaml_content = yaml.safe_load(f)
    misc = yaml_content.pop("misc")
    setting = PhlowerSetting.model_validate(yaml_content)
    setting = PhlowerSetting.model_validate(
        setting.model_dump()
        | {
            "training": {
                "batch_size": batch_size,
            }
        }
    )

    dataset = OnMemoryPhlowerDataSet.create(
        input_settings=setting.model.inputs,
        label_settings=setting.model.labels,
        field_settings=setting.model.fields,
        directories=directories,
    )
    builder = DataLoaderBuilder.from_setting(setting.training)

    dataloader = builder.create(dataset, drop_last=True)

    tests = misc["tests"]["desired_n_points"]

    for item in dataloader:
        item: LumpedTensorData
        assert len(item.data_directories) == batch_size
        assert item.n_data == batch_size

        for name in item.x_data.keys():
            assert (
                item.x_data[name].n_vertices()
                == int(tests["inputs"][name]) * batch_size
            )

        for name in item.y_data.keys():
            assert (
                item.y_data[name].n_vertices()
                == int(tests["labels"][name]) * batch_size
            )


# endregion
