"""Global test fixtures definitions."""

# Taken from tedi and guid_tracker
from __future__ import annotations

import asyncio
import datetime
import os
import posixpath
import typing

from collections.abc import Iterable, Iterator
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING

from _pytest.monkeypatch import MonkeyPatch

import pytest


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch


import concurrent.futures.thread
import copy
import functools
import glob
import os
import re
import shutil
import sys

from concurrent.futures import Executor, Future
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import discord as dc
import discord.ext.test as dpytest
import pytest_asyncio

from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from discord.client import _LoopSentinel
from discord.ext import commands
from goob_ai.goob_bot import AsyncGoobBot
from requests_toolbelt.multipart import decoder
from vcr import filters

import pytest


# """
# Log levels
# __________

# Setting logging levels during tests with pytest is tricky, since pytest replaces your logging configuration with its
# own. To help with this, we leverage our TestContext fixture, which is used by most of our tests.
# By default, it sets the log level to WARNING, unless we are running under a debugger, in which case it defaults to
# DEBUG.

# You can also explicitly set the log level for a specific test with the `set_log_level()` method of TestContext.

# When not running a test, Venice code defaults the log level in the same pattern: DEBUG if running under a debugger
# and WARNING otherwise. This default can be overriden by setting the environment variable VENICE_LOGLEVEL.

# """

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


########################################## vcr ##########################################
@dataclass
class TestContext:
    __test__ = False

    data_path: Path
    out_path: Path
    caplog: Optional[LogCaptureFixture]

    def __post_init__(self) -> None:
        if self.caplog:
            self.caplog.set_level("DEBUG" if sys.gettrace() is not None else "WARNING")

    def set_log_level(self, level: str | int) -> None:
        if not self.caplog:
            raise RuntimeError("No caplog")
        self.caplog.set_level(level)


def patch_env(key: str, fake_value: str, monkeypatch: MonkeyPatch) -> None:
    if not os.environ.get(key):
        monkeypatch.setenv(key, fake_value)


def is_opensearch_uri(uri: str) -> bool:
    return any(x in uri for x in ["opensearch", "es.amazonaws.com"])


def is_llm_uri(uri: str) -> bool:
    return any(x in uri for x in ["openai", "llm-proxy", "anthropic", "localhost", "127.0.0.1"])


def is_chroma_uri(uri: str) -> bool:
    return any(x in uri for x in ["localhost", "127.0.0.1"])


def request_matcher(r1, r2):
    """
    Custom matcher to determine if the requests are the same
    - For sensei requests, we match the parts of the multipart request. This is needed as we can't compare the body
        directly as the chunk boundary is generated randomly
    - For opensearch requests, we just match the body
    - For openai, allow llm-proxy
    - For others, we match both uri and body
    """
    import rich

    rich.inspect(r1, all=True)
    rich.inspect(r2, all=True)

    if r1.uri == r2.uri:
        if r1.body == r2.body:
            return True
        # elif 'sensei' in r1.uri:
        #     r1_parts = [r1p.content for r1p in decoder.MultipartDecoder(r1.body, r1.headers['Content-Type']).parts]
        #     r2_parts = [r2p.content for r2p in decoder.MultipartDecoder(r2.body, r2.headers['Content-Type']).parts]

        #     parts_same = [r1p == r2p for (r1p, r2p) in zip(r1_parts, r2_parts)]
        #     return all(parts_same)
    elif is_opensearch_uri(r1.uri) and is_opensearch_uri(r2.uri):
        return r1.body == r2.body
    elif is_llm_uri(r1.uri) and is_llm_uri(r2.uri):
        return r1.body == r2.body
    elif is_chroma_uri(r1.uri) and is_chroma_uri(r2.uri):
        return r1.body == r2.body

    return False


# @pytest.fixture(scope="module")
# def vcr(vcr):
#     # vcr.register_matcher("request_matcher", request_matcher)
#     # vcr.match_on = ["request_matcher"]
#     return vcr


# SOURCE: https://github.com/kiwicom/pytest-recording/tree/master
def pytest_recording_configure(config, vcr):
    vcr.register_matcher("request_matcher", request_matcher)
    vcr.match_on = ["request_matcher"]


def filter_response(response):
    """
    If the response has a 'retry-after' header, we set it to 0 to avoid waiting for the retry time
    """

    if "retry-after" in response["headers"]:
        response["headers"]["retry-after"] = "0"

    return response


def filter_request(request):
    """
    If the request is of type multipart/form-data we don't filter anything, else we perform two additional filterings -
    1. Processes the request body text, replacing today's date with a placeholder.This is necessary to ensure
        consistency in recorded VCR requests. Without this modification, requests would contain varying body text
        with older dates, leading to failures in request body text comparison when executed with new dates.
    2. Filter out specific fields from post data fields
    """

    # vcr does not handle multipart/form-data correctly as reported on https://github.com/kevin1024/vcrpy/issues/521
    # so let it pass through as is
    if ctype := request.headers.get("Content-Type"):
        ctype = ctype.decode("utf-8") if isinstance(ctype, bytes) else ctype
        if "multipart/form-data" in ctype:
            return request

    request = copy.deepcopy(request)

    if ".tiktoken" in request.path:
        # request to https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
        # can be made by ChatLLMInvoker of venice-gentech
        return None

    # filter dates
    if request.body is not None:
        dummy_date_str = "today's date which is DUMMY_DATE"
        request_body_str = request.body.decode("utf-8")
        matches = re.findall(r"today's date which is \d{4}-\d{2}-\d{2}", request_body_str)
        if len(matches) > 0:
            for match in matches:
                request_body_str = request_body_str.replace(match, dummy_date_str)
            request.body = request_body_str.encode("utf-8")

    # filter fields from post
    filter_post_data_parameters = ["api-version", "client_id", "client_secret", "code", "username", "password"]
    replacements = [p if isinstance(p, tuple) else (p, None) for p in filter_post_data_parameters]
    filter_function = functools.partial(filters.replace_post_data_parameters, replacements=replacements)
    request = filter_function(request)

    return request


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            ("authorization", "DUMMY_AUTHORIZATION"),
            # ("Set-Cookie", "DUMMY_COOKIE"),
            ("x-api-key", "DUMMY_API_KEY"),
            ("api-key", "DUMMY_API_KEY"),
        ],
        "ignore_localhost": False,
        # "record_mode": "once",
        # "filter_query_parameters": ["api-version", "client_id", "client_secret", "code"],
        "filter_query_parameters": ["api-version", "client_id", "client_secret", "code", "api_key"],
        "before_record_request": filter_request,
        "before_record_response": filter_response,
        # !! DO NOT filter post data via a config here, but add it to the filter_data function above !!
        # We don't match requests on 'headers' and 'host' since they vary a lot
        # We tried not matching on 'body' since a POST request - specifically a request to the extract service - appears
        # to have some differences in the body - but that didn't work. Then we didn't get a match at all. So left as is.
        # See https://vcrpy.readthedocs.io/en/latest/configuration.html#request-matching
        # "match_on": ["method", "scheme", "port", "path", "query", "body", "uri"],
        "match_on": ["method", "scheme", "port", "path", "query", "body"],
    }


# @pytest.fixture(scope='module')
# def vcr_cassette_dir(request):
#     # Put all cassettes in vhs/{module}/{test}.yaml
#     return os.path.join('vhs', request.module.__name__)
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
