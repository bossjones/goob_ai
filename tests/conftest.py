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


import glob
import os

import discord as dc
import discord.ext.test as dpytest
import pytest_asyncio

from discord.client import _LoopSentinel
from discord.ext import commands
from goob_ai.goob_bot import AsyncGoobBot

import pytest


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


# # ---------------------------------------------------------------
# # SOURCE: https://github.com/Zorua162/dpystest_minimal/blob/ebbe7f61c741498b8ea8897fc22a11781e4d67bf/conftest.py#L4
# # ---------------------------------------------------------------
# @pytest_asyncio.fixture
# async def bot(event_loop):
#     """Initialise bot to be able to run tests on it"""
#     # Create the bot, similar to how it is done in start_bot
#     bot = Bot(event_loop)
#     bot.add_command(ping)
#     bot.add_command(create_channel)
#     bot.add_command(get_channel)
#     bot.add_command(get_channel_history)

#     if isinstance(bot.loop, _LoopSentinel):
#         await bot._async_setup_hook()

#     # Configure the bot to be in a test environment (similar to bot.run)
#     dpytest.configure(bot)
#     await bot.setup_hook()
#     assert dpytest.get_message().content == "Message from setup hook"

#     return bot

# def pytest_sessionfinish():
#     """Clean up files"""
#     files = glob.glob('./dpytest_*.dat')
#     for path in files:
#         try:
#             os.remove(path)
#         except Exception as e:
#             print(f"Error while deleting file {path}: {e}")
# # ---------------------------------------------------------------


def pytest_sessionfinish(session, exitstatus):
    """Code to execute after all tests."""

    # dat files are created when using attachements
    print("\n-------------------------\nClean dpytest_*.dat files")
    fileList = glob.glob("./dpytest_*.dat")
    for filePath in fileList:
        try:
            os.remove(filePath)
        except Exception:
            print("Error while deleting file : ", filePath)
