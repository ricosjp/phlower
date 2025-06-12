import pathlib

import phlower
import tomli


def test__version_same_as_pyproject_toml():
    pyproject_path = pathlib.Path(__file__) / "../../pyproject.toml"
    with open(pyproject_path.resolve(), "rb") as fr:
        pyproject_data = tomli.load(fr)

    version_number = pyproject_data["project"]["version"]
    assert phlower.__version__ == version_number
