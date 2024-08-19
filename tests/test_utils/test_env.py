from unittest import mock

import pytest
from phlower.utils import determine_n_process


@pytest.mark.parametrize(
    "n_cpu, max_process, desired",
    [(10, 3, 3), (1, 3, 1), (10, None, 10), (0, None, 0)],
)
def test__determine_n_process(
    n_cpu: int, max_process: int | None, desired: int
):
    with mock.patch("os.cpu_count", return_value=n_cpu):
        n_process = determine_n_process(max_process)

        assert n_process == desired
