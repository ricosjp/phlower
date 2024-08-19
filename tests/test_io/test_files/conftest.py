import pathlib
import shutil

import pytest


@pytest.fixture(scope="module")
def setup_test_dir() -> pathlib.Path:
    directory = pathlib.Path(__file__).parent
    test_dir = directory / "tmp"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    return test_dir
