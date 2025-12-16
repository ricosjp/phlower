import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=["cpu", "cuda:0"],
        help="Device to run tests on: 'cpu' or 'cuda:0'. Default is 'cpu'.",
    )


@pytest.fixture
def auto_device(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("device")
