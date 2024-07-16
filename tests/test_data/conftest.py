import pathlib
import shutil
from collections import defaultdict

import numpy as np
import pytest
import scipy.sparse as sp

from phlower.io import PhlowerNumpyFile
from phlower.utils.typing import ArrayDataType
from phlower.data import LazyPhlowerDataset

_output_base_directory = pathlib.Path(__file__).parent / "tmp/datasets"


@pytest.fixture
def output_base_directory():
    return _output_base_directory


@pytest.fixture(scope="module")
def create_tmp_dataset():
    if _output_base_directory.exists():
        shutil.rmtree(_output_base_directory)
    _output_base_directory.mkdir(parents=True)

    directory_names = ["data0", "data1", "data2"]
    name2dense_shape: dict[str, tuple[int, ...]] = {
        "x0": (1, 3, 4),
        "x1": (10, 5),
        "x2": (11, 3),
        "y0": (1, 3, 4),
    }
    name2sparse_shape: dict[str, tuple[int, ...]] = {
        "s0": (5, 5),
        "s1": (10, 5),
    }

    results: dict[str, dict[str, ArrayDataType]] = defaultdict(dict)
    for name in directory_names:
        _output_directory = _output_base_directory / name
        _output_directory.mkdir()

        for v_name, v_shape in name2dense_shape.items():
            arr = np.random.rand(*v_shape)
            PhlowerNumpyFile.save(_output_directory, v_name, arr)
            results[name][v_name] = arr

        for v_name, v_shape in name2sparse_shape.items():
            arr = sp.random(*v_shape, density=0.1)
            PhlowerNumpyFile.save(_output_directory, v_name, arr)
            results[name][v_name] = arr

    return results

