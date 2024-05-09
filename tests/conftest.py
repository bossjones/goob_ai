"""
Global test fixtures definitions.
"""

# Taken from tedi and guid_tracker
from __future__ import annotations

import asyncio
import datetime
import os
import posixpath
import typing

from pathlib import Path, PosixPath
from typing import TYPE_CHECKING, Iterable, Iterator

from _pytest.monkeypatch import MonkeyPatch

import pytest


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

# from prisma.utils import get_or_create_event_loop
# from prisma.testing import reset_client

# from ._utils import request_has_client

# SOURCE: https://github.com/RobertCraigie/prisma-client-py/blob/da53c4280756f1a9bddc3407aa3b5f296aa8cc10/lib/testing/shared_conftest/_shared_conftest.py#L26-L30
# @pytest.fixture(scope='session')
# def event_loop() -> Iterable[asyncio.AbstractEventLoop]:
#     loop = get_or_create_event_loop()
#     yield loop
#     loop.close()


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
