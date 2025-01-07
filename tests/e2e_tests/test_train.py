import pathlib
import random
import shutil

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from phlower import PhlowerTensor
from phlower.io import PhlowerDirectory
from phlower.services.predictor import PhlowerPredictor
from phlower.services.trainer import PhlowerTrainer
from phlower.settings import PhlowerSetting

_OUTPUT_DIR = pathlib.Path(__file__).parent / "_tmp"


@pytest.fixture(scope="module")
def prepare_sample_preprocessed_files():
    random.seed(11)
    np.random.seed(11)
    if _OUTPUT_DIR.exists():
        shutil.rmtree(_OUTPUT_DIR)
    _OUTPUT_DIR.mkdir()

    base_preprocessed_dir = _OUTPUT_DIR / "preprocessed"
    base_preprocessed_dir.mkdir()

    n_cases = 3
    dtype = np.float32
    for i in range(n_cases):
        n_nodes = 100 * (i + 1)
        preprocessed_dir = base_preprocessed_dir / f"case_{i}"
        preprocessed_dir.mkdir()

        nodal_initial_u = np.random.rand(n_nodes, 3, 1)
        np.save(
            preprocessed_dir / "nodal_initial_u.npy",
            nodal_initial_u.astype(dtype),
        )

        # nodal_last_u = np.random.rand(n_nodes, 3, 1)
        np.save(
            preprocessed_dir / "nodal_last_u.npy", nodal_initial_u.astype(dtype)
        )

        rng = np.random.default_rng()
        nodal_nadj = sp.random(n_nodes, n_nodes, density=0.1, random_state=rng)
        sp.save_npz(
            preprocessed_dir / "nodal_nadj", nodal_nadj.tocoo().astype(dtype)
        )

        (preprocessed_dir / "preprocessed").touch()


@pytest.fixture(scope="module")
def simple_training(prepare_sample_preprocessed_files: None) -> PhlowerTensor:
    phlower_path = PhlowerDirectory(_OUTPUT_DIR)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml("tests/e2e_tests/data/train.yml")

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = _OUTPUT_DIR / "model"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    loss = trainer.train(
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
        output_directory=output_directory,
    )
    return loss


@pytest.mark.e2e_test
def test__training_with_multiple_batch_size(
    prepare_sample_preprocessed_files: None,
):
    phlower_path = PhlowerDirectory(_OUTPUT_DIR)

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
    output_directory = _OUTPUT_DIR / "model_batch_size"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    loss = trainer.train(
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
        output_directory=output_directory,
    )
    assert loss.has_dimension
    assert not torch.isinf(loss.to_tensor())
    assert not torch.isnan(loss.to_tensor())


@pytest.mark.e2e_test
def test__training_with_multiple_batch_size_with_gpu(
    prepare_sample_preprocessed_files: None,
):
    phlower_path = PhlowerDirectory(_OUTPUT_DIR)

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
    output_directory = _OUTPUT_DIR / "model_batch_size_gpu"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    loss = trainer.train(
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
        output_directory=output_directory,
    )
    assert loss.has_dimension
    assert not torch.isinf(loss.to_tensor())
    assert not torch.isnan(loss.to_tensor())


def test__same_loss_value_when_traing_data_and_validation_data_is_same(
    prepare_sample_preprocessed_files: None,
):
    phlower_path = PhlowerDirectory(_OUTPUT_DIR)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml("tests/e2e_tests/data/train.yml")

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = _OUTPUT_DIR / "model_one_data"
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


@pytest.mark.e2e_test
def test__simple_training(simple_training: PhlowerTensor):
    loss: PhlowerTensor = simple_training

    assert loss.has_dimension
    assert not torch.isinf(loss.to_tensor())
    assert not torch.isnan(loss.to_tensor())


@pytest.mark.e2e_test
def test__predict(simple_training: PhlowerTensor):
    setting = PhlowerSetting.read_yaml("tests/e2e_tests/data/predict.yml")
    model_directory = _OUTPUT_DIR / "model"

    predictor = PhlowerPredictor(
        model_directory=model_directory, predict_setting=setting.prediction
    )
    phlower_path = PhlowerDirectory(_OUTPUT_DIR)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    for result in predictor.predict(preprocessed_directories):
        for k in result.keys():
            print(f"{k}: {result[k].dimension}")
