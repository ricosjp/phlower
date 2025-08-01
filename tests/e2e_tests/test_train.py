import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest
from phlower.io import PhlowerDirectory
from phlower.services.trainer import PhlowerTrainer
from phlower.settings import PhlowerSetting


@pytest.mark.e2e_test
def test__training_with_multiple_batch_size(
    prepare_sample_preprocessed_files_fixture: pathlib.Path,
):
    output_dir = prepare_sample_preprocessed_files_fixture
    phlower_path = PhlowerDirectory(output_dir / "preprocessed")

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml(
        "tests/e2e_tests/data/train_batch_size.yml"
    )
    assert setting.training.batch_size > 1

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = output_dir / "model_batch_size"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    loss = trainer.train(
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
        output_directory=output_directory,
    )
    assert loss > 0


@pytest.mark.e2e_test
def test__training_with_multiple_batch_size_with_gpu(
    prepare_sample_preprocessed_files_fixture: pathlib.Path,
):
    output_dir = prepare_sample_preprocessed_files_fixture
    phlower_path = PhlowerDirectory(output_dir / "preprocessed")

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml(
        "tests/e2e_tests/data/train_batch_size_gpu.yml"
    )
    assert setting.training.batch_size > 1

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = output_dir / "model_batch_size_gpu"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    loss = trainer.train(
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
        output_directory=output_directory,
    )
    assert loss > 0


@pytest.mark.e2e_test
def test__same_loss_value_when_traing_data_and_validation_data_is_same(
    prepare_sample_preprocessed_files_fixture: None,
):
    output_dir = prepare_sample_preprocessed_files_fixture
    phlower_path = PhlowerDirectory(output_dir / "preprocessed")

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml("tests/e2e_tests/data/train.yml")

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = output_dir / "model_one_data"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    _ = trainer.train(
        train_directories=[preprocessed_directories[0]],
        validation_directories=[preprocessed_directories[0]],
        output_directory=output_directory,
    )

    df = pd.read_csv(output_directory / "log.csv")

    train_loss = df.loc[:, "train_loss"].to_numpy()
    validation_loss = df.loc[:, "validation_loss"].to_numpy()

    np.testing.assert_array_almost_equal(train_loss, validation_loss)
