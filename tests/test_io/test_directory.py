import pathlib
import shutil

import pytest

from phlower.io import PhlowerDirectory

TEST_DATA_DIR = pathlib.Path(__file__).parent / "tmp"


@pytest.fixture(scope="module")
def create_test_cases():
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)
    TEST_DATA_DIR.mkdir()

    (TEST_DATA_DIR / "variable1.npy").touch()
    (TEST_DATA_DIR / "variable2.npz.enc").touch()


@pytest.mark.parametrize(
    "path", ["tests/data/deform", "tests/data/csv_prepost/raw"]
)
def test__initialize(path: str):
    phlower_dir = PhlowerDirectory(pathlib.Path(path))

    assert phlower_dir.path == pathlib.Path(path)


@pytest.mark.parametrize(
    "variable_name, ext", [("variable1", ".npy"), ("variable2", ".npz.enc")]
)
def test__find_variable_file(
    variable_name: str, ext: str, create_test_cases: None
):
    phlower_dir = PhlowerDirectory(TEST_DATA_DIR)

    phlower_file = phlower_dir.find_variable_file(variable_name)
    assert phlower_file.file_path == TEST_DATA_DIR / f"{variable_name}{ext}"
    assert phlower_dir.exist_variable_file(variable_name)


@pytest.mark.parametrize(
    "variable_name, ext", [("variable3", ".npz"), ("variable4", ".npz.enc")]
)
def test__failed_find_variable_file(
    variable_name: str, ext: str, create_test_cases: None
):
    phlower_dir = PhlowerDirectory(TEST_DATA_DIR)

    assert phlower_dir.exist_variable_file(variable_name) is False
    with pytest.raises(ValueError):
        _ = phlower_dir.find_variable_file(variable_name)


@pytest.mark.parametrize(
    "variable_name, ext", [("variable3", ".npz"), ("variable4", ".npz.enc")]
)
def test__find_variable_file_as_None(
    variable_name: str, ext: str, create_test_cases: None
):
    phlower_dir = PhlowerDirectory(TEST_DATA_DIR)

    phlower_file = phlower_dir.find_variable_file(
        variable_name, allow_missing=True
    )
    assert phlower_file is None
    assert phlower_dir.exist_variable_file(variable_name) is False
