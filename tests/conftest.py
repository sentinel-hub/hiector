import asyncio
import os
import shutil
import subprocess
import time

import _pytest
import pytest

ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = os.path.join(ROOT, "_data")


def module_relpath(request: _pytest.fixtures.FixtureRequest, folder_name: str = ""):
    """Constructs path for the test execution from the test file's name, which it gets from
    pytest.FixtureRequest (https://docs.pytest.org/en/latest/reference.html#request).
    """
    test_name = request.fspath.basename.split(".")[0]
    relpath = os.path.relpath(str(request.fspath.dirname), ROOT)
    return os.path.join(relpath, test_name, folder_name)


def function_relpath(request: _pytest.fixtures.FixtureRequest, folder_name: str = ""):
    """Constructs path for the test execution from the test file's name and function, which it gets from
    pytest.FixtureRequest (https://docs.pytest.org/en/latest/reference.html#request).
    """
    mod_path = module_relpath(request, folder_name=folder_name)
    return os.path.join(mod_path, request.function.__name__)


# --------------------------------- PATH FIXTURES ---------------------------------


@pytest.fixture(scope="module")
def input_folder(request: _pytest.fixtures.FixtureRequest):
    """Returns the input folder path `test/_data/test_name/input`."""
    return os.path.join(DATA_ROOT, module_relpath(request, folder_name="input"))


@pytest.fixture(scope="function")
def output_folder(request: _pytest.fixtures.FixtureRequest):
    """Creates the output folder path `test/_data/test_name/output`.

    It also cleans the output folder before the test runs.
    """

    out_path = os.path.join(DATA_ROOT, function_relpath(request, folder_name="output"))
    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    os.makedirs(out_path)

    yield out_path
