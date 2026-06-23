import pathlib
import random
import shutil

import numpy as np
import pytest
import scipy.sparse as sp
import yaml

from phlower.io import PhlowerDirectory
from phlower.services.trainer import PhlowerContinueTrainer
from phlower.services.trainer._continue_trainer import (
    ContinueState,
    StateUpdater,
)
from phlower.settings import PhlowerSetting

_OUT_CONTINUE_DIR = pathlib.Path(__file__).parent / "_out_continue_training"
_SETTING_DIR = pathlib.Path(__file__).parent / "data/continue_trainer"


def _initialize_tmp_output_directory(output_dir: pathlib.Path):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with (output_dir / ".gitignore").open("w") as fw:
        fw.write("*")


def create_sample_preprocessed_data(output_base_dir: pathlib.Path):
    random.seed(11)
    np.random.seed(11)
    _initialize_tmp_output_directory(output_base_dir)

    base_preprocessed_dir = output_base_dir / "preprocessed"
    base_preprocessed_dir.mkdir()

    n_cases = 4
    dtype = np.float32
    for i in range(n_cases):
        n_nodes = 100 * (i + 1)
        preprocessed_dir = base_preprocessed_dir / f"case_{i}"
        preprocessed_dir.mkdir(exist_ok=True)

        nodal_initial_u = np.random.rand(n_nodes, 3, 1)
        np.save(
            preprocessed_dir / "nodal_initial_u.npy",
            nodal_initial_u.astype(dtype),
        )

        nodal_last_u = np.random.rand(n_nodes, 3, 1)
        np.save(
            preprocessed_dir / "nodal_last_u.npy", nodal_last_u.astype(dtype)
        )

        n_time_series = 5
        nodal_last_ts_u = np.random.rand(n_time_series, n_nodes, 3, 1)
        np.save(
            preprocessed_dir / "nodal_last_ts_u.npy",
            nodal_last_ts_u.astype(dtype),
        )

        nodal_initial_p = np.random.rand(n_nodes, 1)
        np.save(
            preprocessed_dir / "nodal_initial_p.npy",
            nodal_initial_p.astype(dtype),
        )
        nodal_last_p = np.random.rand(n_nodes, 1)
        np.save(
            preprocessed_dir / "nodal_last_p.npy", nodal_last_p.astype(dtype)
        )

        rng = np.random.default_rng()
        nodal_nadj = sp.random(n_nodes, n_nodes, density=0.1, random_state=rng)
        sp.save_npz(
            preprocessed_dir / "nodal_nadj", nodal_nadj.tocoo().astype(dtype)
        )

        (preprocessed_dir / "preprocessed").touch()


def _update_training_setting(
    setting: PhlowerSetting, update_dict: dict
) -> PhlowerSetting:
    return PhlowerSetting.model_validate(
        setting.model_dump()
        | {"training": setting.training.model_dump() | update_dict}
    )


@pytest.fixture(scope="module")
def prepare_sample_preprocessed_files():
    create_sample_preprocessed_data(_OUT_CONTINUE_DIR)


def _train(yaml_filename: str, output_base_dir: pathlib.Path):
    phlower_path = PhlowerDirectory(_OUT_CONTINUE_DIR)
    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    _initialize_tmp_output_directory(output_base_dir)

    setting = PhlowerSetting.read_yaml(_SETTING_DIR / yaml_filename)
    trainer = PhlowerContinueTrainer.from_setting(setting=setting)

    trainer.train(
        output_directory=output_base_dir / "model",
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
    )


@pytest.mark.parametrize(
    "yaml_filename",
    [
        "train_wo_continue.yml",
    ],
)
def test__train_once_if_continue_setting_is_missing(
    yaml_filename: str, prepare_sample_preprocessed_files: None
):
    output_base_dir = _OUT_CONTINUE_DIR / "train_once"
    _train(yaml_filename, output_base_dir)

    model_dirs = [
        p
        for p in output_base_dir.iterdir()
        if p.is_dir() and p.name.startswith("model")
    ]
    assert len(model_dirs) == 1


@pytest.mark.parametrize(
    "yaml_filename",
    [
        "train_cont_3_w_lr_decay.yml",
        "train_cont_3_w_optim_switch.yml",
    ],
)
def test__train_cont3(
    yaml_filename: str, prepare_sample_preprocessed_files: None
):
    output_base_dir = _OUT_CONTINUE_DIR / yaml_filename.split(".")[0]
    _train(yaml_filename, output_base_dir)

    model_dirs = [
        p
        for p in output_base_dir.iterdir()
        if p.is_dir() and p.name.startswith("model")
    ]
    assert len(model_dirs) == 4

    # check the dumped continue_state.yml
    for i in range(3):
        continue_state_file = list(
            output_base_dir.glob(f"model_cont{i + 1}*/continue_state.yml")
        )
        assert len(continue_state_file) == 1
        continue_state_file = continue_state_file[0]
        assert continue_state_file.exists()
        with continue_state_file.open("r") as fr:
            content = yaml.safe_load(fr.read())
        assert content["continue_count"] == i + 1


# region test for restart


@pytest.mark.parametrize(
    "yaml_filename",
    [
        "train_wo_continue.yml",
    ],
)
def test__restart_first_attempt(
    yaml_filename: str, prepare_sample_preprocessed_files: None
):
    output_base_dir = _OUT_CONTINUE_DIR / "restart_first_attempt"
    _train(yaml_filename, output_base_dir)

    # --- Overwrite setting file to mock the stopped training
    setting = PhlowerSetting.read_yaml(output_base_dir / "model/model.yml")
    setting = _update_training_setting(
        setting,
        {
            "n_epoch": 5,
            "continue_setting": {
                "stop_count": 2,
                "continue_type": "lr_decay",
                "parameters": {"lr_factor": 0.5},
            },
        },
    )
    with open(output_base_dir / "model/model.yml", "w") as fw:
        yaml.safe_dump(setting.model_dump(), fw)

    # ---

    n_cont = len(list(output_base_dir.glob("model_cont*")))
    assert n_cont == 0

    # Restart the training

    trainer = PhlowerContinueTrainer.restart_from(output_base_dir / "model")
    phlower_path = PhlowerDirectory(_OUT_CONTINUE_DIR)
    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )
    trainer.train(
        output_directory=output_base_dir / "model",
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
    )

    n_epochs = len(list(output_base_dir.glob("model/weights/*.pth")))
    assert n_epochs == 5

    n_cont = len(list(output_base_dir.glob("model_cont*")))
    assert n_cont == 2


@pytest.mark.parametrize(
    "yaml_filename",
    [
        "train_cont_3_w_lr_decay.yml",
    ],
)
def test__restart_continue_attempt(
    yaml_filename: str, prepare_sample_preprocessed_files: None
):
    output_base_dir = _OUT_CONTINUE_DIR / "restart_cont_attempt"
    _train(yaml_filename, output_base_dir)

    cont_dirs = sorted(output_base_dir.glob("model_cont*"))
    n_cont = len(cont_dirs)
    assert n_cont == 3

    # --- Overwrite setting file to mock the stopped training
    setting = PhlowerSetting.read_yaml(cont_dirs[-1] / "model.yml")
    setting = _update_training_setting(
        setting,
        {
            "n_epoch": 5,
            "continue_setting": {
                "stop_count": 5,
                "continue_type": "lr_decay",
                "parameters": {"lr_factor": 0.5},
            },
        },
    )
    with open(cont_dirs[-1] / "model.yml", "w") as fw:
        yaml.safe_dump(setting.model_dump(), fw)

    # ---

    # Restart the training

    trainer = PhlowerContinueTrainer.restart_from(cont_dirs[-1])
    phlower_path = PhlowerDirectory(_OUT_CONTINUE_DIR)
    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )
    trainer.train(
        output_directory=cont_dirs[-1],
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
    )

    n_cont = len(list(output_base_dir.glob("model_cont*")))
    assert n_cont == 5


@pytest.mark.parametrize(
    "yaml_filename",
    [
        "train_wo_continue.yml",
    ],
)
def test__restart_first_attempt_when_output_directory_is_not_the_same(
    yaml_filename: str, prepare_sample_preprocessed_files: None
):
    output_base_dir = _OUT_CONTINUE_DIR / "restart_first_attempt"
    _train(yaml_filename, output_base_dir)

    # --- Overwrite setting file to mock the stopped training
    setting = PhlowerSetting.read_yaml(output_base_dir / "model/model.yml")
    setting = _update_training_setting(
        setting,
        {
            "n_epoch": 5,
            "continue_setting": {
                "stop_count": 2,
                "continue_type": "lr_decay",
                "parameters": {"lr_factor": 0.5},
            },
        },
    )
    with open(output_base_dir / "model/model.yml", "w") as fw:
        yaml.safe_dump(setting.model_dump(), fw)

    # ---

    n_cont = len(list(output_base_dir.glob("model_cont*")))
    assert n_cont == 0

    # Restart the training

    new_output_dir = output_base_dir.parent / "restart_first_attempt_new"
    trainer = PhlowerContinueTrainer.restart_from(output_base_dir / "model")
    phlower_path = PhlowerDirectory(_OUT_CONTINUE_DIR)
    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )
    trainer.train(
        output_directory=new_output_dir / "model",
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
    )

    n_epochs = len(list(new_output_dir.glob("model/weights/*.pth")))
    assert n_epochs == 2  # 5 - 3 = 2


@pytest.mark.parametrize(
    "yaml_filename",
    [
        "train_cont_3_w_lr_decay.yml",
    ],
)
def test__restart_continue_attempt_when_output_directory_is_not_the_same(
    yaml_filename: str, prepare_sample_preprocessed_files: None
):
    output_base_dir = _OUT_CONTINUE_DIR / "restart_cont_attempt"
    _train(yaml_filename, output_base_dir)

    cont_dirs = sorted(output_base_dir.glob("model_cont*"))
    n_cont = len(cont_dirs)
    assert n_cont == 3

    # --- Overwrite setting file to mock the stopped training
    setting = PhlowerSetting.read_yaml(cont_dirs[-1] / "model.yml")
    setting = _update_training_setting(
        setting,
        {
            "n_epoch": 5,
            "continue_setting": {
                "stop_count": 5,
                "continue_type": "lr_decay",
                "parameters": {"lr_factor": 0.5},
            },
        },
    )
    with open(cont_dirs[-1] / "model.yml", "w") as fw:
        yaml.safe_dump(setting.model_dump(), fw)

    # ---

    # Restart the training

    output_base_dir_new = _OUT_CONTINUE_DIR / "restart_cont_attempt_new"
    trainer = PhlowerContinueTrainer.restart_from(cont_dirs[-1])
    phlower_path = PhlowerDirectory(_OUT_CONTINUE_DIR)
    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )
    trainer.train(
        output_directory=output_base_dir_new / "model",
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
    )

    n_cont = len(list(output_base_dir_new.glob("model_cont*")))
    assert n_cont == 2  # 1 cont starts from the intermediate, 2 cont are new


# endregion


# region unit tests for StateGenerator


@pytest.mark.parametrize(
    "yaml_filename",
    [
        "train_cont_3_w_lr_decay.yml",
        "train_cont_3_w_optim_switch.yml",
    ],
)
@pytest.mark.parametrize("current_count", [1, 2])
def test__updated_state_for_common_items(
    yaml_filename: str, current_count: int, tmp_path: pathlib.Path
):
    setting = PhlowerSetting.read_yaml(_SETTING_DIR / yaml_filename)
    cont_setting = setting.training.continue_setting

    state_updator = StateUpdater(tmp_path, cont_setting)
    state = ContinueState(
        current_count=current_count, output_directory=tmp_path, setting=setting
    )

    next_state = state_updator.update(state)

    assert next_state.current_count == current_count + 1
    next_training = next_state.setting.training
    assert next_training.initializer_setting.type_name == "pretrained"
    assert next_training.initializer_setting.reference_directory == tmp_path


# endregion
