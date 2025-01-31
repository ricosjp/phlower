import pathlib
import random
import shutil

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from phlower.io import select_snapshot_file
from phlower.io._model_selector import ModelSelectorBuilder
from phlower.utils.enums import ModelSelectionType

TEST_DATA_DIR = pathlib.Path(__file__).parent / "tmp"


@given(st.sampled_from(ModelSelectionType))
def test__selctor_builder(select_type: ModelSelectionType):
    _ = ModelSelectorBuilder.create(select_type.value)


@pytest.mark.parametrize("selection_name", ["best_of_best", "none"])
def test__not_implemented_selection_name(selection_name: str):
    with pytest.raises(NotImplementedError):
        _ = ModelSelectorBuilder.create(selection_name)


@pytest.fixture(scope="module")
def prepare_snapshots():
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)

    TEST_DATA_DIR.mkdir()

    for i in range(10):
        file_path = TEST_DATA_DIR / f"snapshot_epoch_{i}.pth"
        file_path.touch()

    epochs = [i + 1 for i in range(i)]
    train_loss = [random.random() for i in range(i)]
    validation_loss = [random.random() for i in range(i)]

    df = pd.DataFrame(
        data={
            "epoch": epochs,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        },
        index=None,
    )
    df.to_csv(TEST_DATA_DIR / "log.csv")


def test__best_select_model(prepare_snapshots: None):
    actual_path = select_snapshot_file(
        TEST_DATA_DIR, selection_mode=ModelSelectionType.BEST.value
    )

    df = pd.read_csv(TEST_DATA_DIR / "log.csv", index_col=None, header=0)
    idx = df.loc[:, "validation_loss"].idxmin()
    epoch = df.loc[idx, "epoch"]

    assert actual_path.epoch == epoch


def test__latest_select_model(prepare_snapshots: None):
    actual_path = select_snapshot_file(
        TEST_DATA_DIR, selection_mode=ModelSelectionType.LATEST.value
    )
    df = pd.read_csv(TEST_DATA_DIR / "log.csv", index_col=None)
    max_epoch = df.loc[:, "epoch"].max()

    assert actual_path.epoch == max_epoch


def test__train_best_select_model(prepare_snapshots: None):
    actual_path = select_snapshot_file(
        TEST_DATA_DIR, selection_mode=ModelSelectionType.TRAIN_BEST.value
    )

    df = pd.read_csv(TEST_DATA_DIR / "log.csv", index_col=None)
    idx = df.loc[:, "train_loss"].idxmin()
    epoch = df.loc[idx, "epoch"]

    assert actual_path.epoch == epoch


@pytest.mark.parametrize("epoch", [1, 5, 6, 8])
def test__spcified_model_selector(epoch: int, prepare_snapshots: None):
    actual_path = select_snapshot_file(
        TEST_DATA_DIR,
        selection_mode=ModelSelectionType.SPECIFIED.value,
        target_epoch=epoch,
    )

    assert actual_path.epoch == epoch


@pytest.mark.parametrize("epoch", [-1, None])
def test__spcified_model_selector_value_error(
    epoch: int, prepare_snapshots: None
):
    with pytest.raises(ValueError):
        _ = select_snapshot_file(
            TEST_DATA_DIR,
            selection_mode=ModelSelectionType.SPECIFIED.value,
            target_epoch=epoch,
        )


@pytest.mark.parametrize("epoch", [100, 200])
def test__spcified_model_selector_not_existed(
    epoch: int, prepare_snapshots: None
):
    with pytest.raises(FileNotFoundError):
        _ = select_snapshot_file(
            TEST_DATA_DIR,
            selection_mode=ModelSelectionType.SPECIFIED.value,
            target_epoch=epoch,
        )
