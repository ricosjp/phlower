import pathlib
import secrets

import pytest

from phlower.io import PhlowerYamlFile

TEST_ENCRYPT_KEY = secrets.token_bytes(32)


@pytest.mark.parametrize(
    "path, ext",
    [("./sample/sample.yml", ".yml"), ("./sample/sample.yml.enc", ".yml.enc")],
)
def test__check_extension_type(path: str, ext: str):
    path = pathlib.Path(path)
    phlower_path = PhlowerYamlFile(path)
    assert phlower_path._ext_type.value == ext
    assert phlower_path.file_path == path


@pytest.mark.parametrize(
    "path", [("./sample/sample.npy"), ("./sample/sample.npy.enc")]
)
def test__check_error_extension_type(path: str):
    path = pathlib.Path(path)
    with pytest.raises(NotImplementedError):
        _ = PhlowerYamlFile(path)


@pytest.mark.parametrize(
    "path, enc",
    [("./sample/sample.yml", False), ("./sample/sample.yml.enc", True)],
)
def test__is_encrypted(path: str, enc: bool):
    path = pathlib.Path(path)
    phlower_path = PhlowerYamlFile(path)
    assert phlower_path.is_encrypted == enc


@pytest.mark.parametrize(
    "encrypt_key, decrypt_key",
    [
        (None, None),
        (None, TEST_ENCRYPT_KEY),
        (TEST_ENCRYPT_KEY, TEST_ENCRYPT_KEY),
    ],
)
def test__save_and_load(
    encrypt_key: bytes | None,
    decrypt_key: bytes | None,
    setup_test_dir: pathlib.Path,
):
    sample_data = {"a": 1, "b": 2}

    saved_path = PhlowerYamlFile.save(
        output_directory=setup_test_dir,
        file_basename="sample",
        data=sample_data,
        encrypt_key=encrypt_key,
        allow_overwrite=True,
    )

    assert saved_path.file_path.exists()

    loaded_data = saved_path.load(decrypt_key=decrypt_key)
    assert loaded_data == sample_data


def test__cannnot_load_encrypt_data_without_key(setup_test_dir: pathlib.Path):
    sample_data = {"a": 1, "b": 2}

    saved_path = PhlowerYamlFile.save(
        output_directory=setup_test_dir,
        file_basename="sample",
        data=sample_data,
        encrypt_key=TEST_ENCRYPT_KEY,
        allow_overwrite=True,
    )

    with pytest.raises(ValueError):
        _ = saved_path.load()


def test__save_not_allowed_overwrite(setup_test_dir: pathlib.Path):
    sample_data = {"a": 1, "b": 2}

    path = setup_test_dir / "sample.yml"
    path.touch(exist_ok=True)

    phlower_path = PhlowerYamlFile(path)
    with pytest.raises(FileExistsError):
        PhlowerYamlFile.save(
            output_directory=setup_test_dir,
            file_basename="sample",
            data=sample_data,
            allow_overwrite=False,
        )
        phlower_path.save(sample_data, overwrite=False)
