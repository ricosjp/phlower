import pathlib
import shutil

import pytest
from phlower.io import PhlowerDirectory
from phlower.services.predictor import PhlowerPredictor
from phlower.services.trainer import PhlowerTrainer
from phlower.settings import PhlowerSetting


@pytest.fixture(scope="module")
def simple_training(
    prepare_sample_preprocessed_files_fixture: pathlib.Path,
) -> pathlib.Path:
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

    _ = trainer.train(
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
        output_directory=output_directory,
    )
    return output_directory


@pytest.mark.e2e_test
def test__predict(simple_training: pathlib.Path):
    setting = PhlowerSetting.read_yaml("tests/e2e_tests/data/predict.yml")
    model_directory = simple_training

    predictor = PhlowerPredictor(
        model_directory=model_directory, predict_setting=setting.prediction
    )
    phlower_path = PhlowerDirectory(model_directory.parent)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    for result in predictor.predict(
        preprocessed_directories, perform_inverse_scaling=False
    ):
        for k in result.prediction_data.keys():
            print(f"{k}: {result.prediction_data[k].dimension}")


@pytest.mark.e2e_test
def test__predict_with_gpu(simple_training: pathlib.Path):
    setting = PhlowerSetting.read_yaml(
        "tests/e2e_tests/data/predict_with_gpu.yml"
    )
    model_directory = simple_training

    predictor = PhlowerPredictor(
        model_directory=model_directory, predict_setting=setting.prediction
    )
    phlower_path = PhlowerDirectory(model_directory.parent)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    for result in predictor.predict(
        preprocessed_directories, perform_inverse_scaling=False
    ):
        for k in result.prediction_data.keys():
            print(f"{k}: {result.prediction_data[k].dimension}")


@pytest.mark.e2e_test
def test__predict_specified(
    simple_training: pathlib.Path,
):
    setting = PhlowerSetting.read_yaml(
        "tests/e2e_tests/data/predict_specified.yml"
    )
    model_directory = simple_training

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

    for result in predictor.predict(
        preprocessed_directories, perform_inverse_scaling=False
    ):
        for k in result.prediction_data.keys():
            print(f"{k}: {result.prediction_data[k].dimension}")
