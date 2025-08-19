import pathlib
import random
import shutil

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from phlower import PhlowerTensor
from phlower.io import PhlowerDirectory, PhlowerNumpyFile
from phlower.services.predictor import PhlowerPredictor
from phlower.services.preprocessing import PhlowerScalingService
from phlower.services.trainer import PhlowerTrainer
from phlower.settings import PhlowerPredictorSetting, PhlowerSetting

DATA_DIR = pathlib.Path(__file__).parent / "data"
OUTPUT_DIR = pathlib.Path(__file__).parent / "_out"


@pytest.fixture(scope="module")
def prepare_sample_preprocessed_files_fixture() -> pathlib.Path:
    output_base_directory = OUTPUT_DIR

    random.seed(11)
    np.random.seed(11)
    if output_base_directory.exists():
        shutil.rmtree(output_base_directory)
    output_base_directory.mkdir()

    base_interim_dir = OUTPUT_DIR / "interim"
    base_interim_dir.mkdir()

    n_cases = 3
    dtype = np.float32
    for i in range(n_cases):
        n_nodes = 100 * (i + 1)
        preprocessed_dir = base_interim_dir / f"case_{i}"
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

        (preprocessed_dir / "converted").touch()

    return output_base_directory


@pytest.fixture(scope="module")
def perform_scaling(
    prepare_sample_preprocessed_files_fixture: pathlib.Path,
) -> pathlib.Path:
    scaling = PhlowerScalingService.from_yaml(DATA_DIR / "simple_train.yml")
    output_base_directory = pathlib.Path(__file__).parent / "_out/preprocessed"

    if output_base_directory.exists():
        shutil.rmtree(output_base_directory)
    output_base_directory.mkdir()

    output_dir = prepare_sample_preprocessed_files_fixture
    phlower_path = PhlowerDirectory(output_dir)

    interim_directories = list(
        phlower_path.find_directory(
            required_filename="converted", recursive=True
        )
    )

    scaling.fit_transform_all(
        interim_data_directories=interim_directories,
        output_base_directory=output_base_directory,
    )
    return output_base_directory


@pytest.fixture(scope="module")
def simple_training(
    perform_scaling: pathlib.Path,
) -> tuple[float, pathlib.Path]:
    phlower_path = PhlowerDirectory(OUTPUT_DIR)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml(DATA_DIR / "simple_train.yml")

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = OUTPUT_DIR / "model"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    loss = trainer.train(
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
        output_directory=output_directory,
    )
    return loss, output_directory


def test__predict_with_inverse_scaling(
    simple_training: tuple[float, pathlib.Path],
):
    _, model_directory = simple_training

    predictor = PhlowerPredictor.from_pathes(
        model_directory=model_directory,
        predict_setting_yaml=DATA_DIR / "predict.yml",
        scaling_setting_yaml=OUTPUT_DIR / "preprocessed/preprocess.yml",
    )
    phlower_path = PhlowerDirectory(model_directory.parent)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    for result in predictor.predict(
        preprocessed_data=preprocessed_directories, perform_inverse_scaling=True
    ):
        for k in result.prediction_data.keys():
            assert isinstance(result.prediction_data[k], np.ndarray)

        for k in result.input_data.keys():
            assert isinstance(result.input_data[k], np.ndarray)


def test__predict_specified(
    simple_training: tuple[float, pathlib.Path],
):
    setting = PhlowerSetting.read_yaml(DATA_DIR / "predict_specified.yml")
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

    for result in predictor.predict(
        preprocessed_directories, perform_inverse_scaling=False
    ):
        for k in result.prediction_data.keys():
            assert isinstance(result.prediction_data[k], PhlowerTensor)
            assert result.prediction_data[k].dimension is not None


def test__predict_with_onmemory_dataset(
    simple_training: tuple[float, pathlib.Path],
):
    _, model_directory = simple_training

    predictor = PhlowerPredictor.from_pathes(
        model_directory=model_directory,
        predict_setting_yaml=DATA_DIR / "predict.yml",
        scaling_setting_yaml=OUTPUT_DIR / "preprocessed/preprocess.yml",
    )
    phlower_path = PhlowerDirectory(model_directory.parent)
    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )
    loaded_data = [
        {
            basename.split(".")[0]: PhlowerNumpyFile(p / basename).load()
            for basename in [
                "nodal_initial_u.npy",
                "nodal_last_u.npy",
                "nodal_nadj.npz",
            ]
        }
        for p in preprocessed_directories
    ]

    for result in predictor.predict(
        preprocessed_data=loaded_data, perform_inverse_scaling=True
    ):
        for k in result.prediction_data.keys():
            assert isinstance(result.prediction_data[k], np.ndarray)


@pytest.mark.parametrize("use_inference_mode", [True, False])
def test__predict_context_manager(
    use_inference_mode: bool, simple_training: tuple[float, pathlib.Path]
):
    _, model_directory = simple_training
    predictor_setting = PhlowerPredictorSetting(
        selection_mode="best", use_inference_mode=use_inference_mode
    )
    predictor = PhlowerPredictor(
        model_directory=model_directory, predict_setting=predictor_setting
    )
    with predictor._predict_context_manager():
        assert torch.is_inference_mode_enabled() == use_inference_mode
        assert not torch.is_grad_enabled()
