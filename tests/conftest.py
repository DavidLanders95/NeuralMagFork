import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "sp: mark a test as a standard problem"
    )
    
def pytest_addoption(parser):
    parser.addoption(
        "--run-sp",
        action="store_true",
        default=False,
        help="Run slow tests",
    )

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-sp"):
        skipper = pytest.mark.skip(reason="Only run when --run-sp is given")
        for item in items:
            if "sp" in item.keywords:
                item.add_marker(skipper)