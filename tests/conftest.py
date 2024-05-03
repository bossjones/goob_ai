"""
Global test fixtures definitions.
"""

# Taken from tedi and guid_tracker

import datetime
import os
from pathlib import PosixPath
import posixpath
import typing

from _pytest.monkeypatch import MonkeyPatch
import pytest

IS_RUNNING_ON_GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTOR"))

HERE = os.path.abspath(os.path.dirname(__file__))
FAKE_TIME = datetime.datetime(2020, 12, 25, 17, 5, 55)

print(f"HERE: {HERE}")

#######################################################################
# only run slow tests when we want to
#######################################################################
# SOURCE: https://doc.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
#######################################################################


@pytest.fixture(name="posixpath_fixture")
def posixpath_fixture(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(os, "path", posixpath)


@pytest.fixture(name="user_homedir")
def user_homedir() -> str:
    return "/Users/runner" if os.environ.get("GITHUB_ACTOR") else "/Users/malcolm"
