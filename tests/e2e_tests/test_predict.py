import pathlib
import shutil

import pytest
import torch
from phlower import PhlowerTensor
from phlower.io import PhlowerDirectory
from phlower.services.predictor import PhlowerPredictor
from phlower.services.trainer import PhlowerTrainer
from phlower.settings import PhlowerSetting


@pytest.fixture(scope="module")
def simple_training(
    prepare_sample_preprocessed_files_fixture: pathlib.Path,
) -> tuple[PhlowerTensor, pathlib.Path]:
    output_dir = prepare_sample_preprocessed_files_fixture
    phlower_path = PhlowerDirectory(output_dir)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml("tests/e2e_tests/data/train.yml")

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = output_dir / "model"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    loss = trainer.train(
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
        output_directory=output_directory,
    )
    return loss, output_directory


@pytest.mark.e2e_test
def test__simple_training(simple_training: tuple[PhlowerTensor, pathlib.Path]):
    loss, _ = simple_training

    assert loss.has_dimension
    assert not torch.isinf(loss.to_tensor())
    assert not torch.isnan(loss.to_tensor())


@pytest.mark.e2e_test
def test__predict(simple_training: tuple[PhlowerTensor, pathlib.Path]):
    setting = PhlowerSetting.read_yaml("tests/e2e_tests/data/predict.yml")
    _, model_directory = simple_training

    predictor = PhlowerPredictor(
        model_directory=model_directory, predict_setting=setting.prediction
    )
    phlower_path = PhlowerDirectory(model_directory.parent)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    for result in predictor.predict(preprocessed_directories):
        for k in result.keys():
            print(f"{k}: {result[k].dimension}")


@pytest.mark.e2e_test
def test__predict_with_gpu(simple_training: tuple[PhlowerTensor, pathlib.Path]):
    setting = PhlowerSetting.read_yaml(
        "tests/e2e_tests/data/predict_with_gpu.yml"
    )
    _, model_directory = simple_training

    predictor = PhlowerPredictor(
        model_directory=model_directory, predict_setting=setting.prediction
    )
    phlower_path = PhlowerDirectory(model_directory.parent)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    for result in predictor.predict(preprocessed_directories):
        for k in result.keys():
            print(f"{k}: {result[k].dimension}")


@pytest.mark.e2e_test
def test__predict_specified(simple_training: PhlowerTensor):
    setting = PhlowerSetting.read_yaml(
        "tests/e2e_tests/data/predict_specified.yml"
    )
    _, model_directory = simple_training

    predictor = PhlowerPredictor(
        model_directory=model_directory, predict_setting=setting.prediction
    )
    phlower_path = PhlowerDirectory(model_directory.parent)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    assert (
        predictor._predict_setting.target_epoch
        == setting.prediction.target_epoch
    )

    for result in predictor.predict(preprocessed_directories):
        for k in result.keys():
            print(f"{k}: {result[k].dimension}")
