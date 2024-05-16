from __future__ import annotations

import asyncio
import os

from asyncio import DefaultEventLoopPolicy
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING, Iterable, Iterator

import pytest_asyncio

from goob_ai import aio_settings
from goob_ai.utils import async_

import pytest


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

IS_RUNNING_ON_GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTOR"))

from pytest_asyncio import is_async_test


def pytest_collection_modifyitems(items):
    for item in items:
        if is_async_test(item):
            pass


# How to test with different event loops
# SOURCE: https://pytest-asyncio.readthedocs.io/en/latest/how-to-guides/multiple_loops.html


# class CustomEventLoopPolicy(DefaultEventLoopPolicy):
#     pass


# @pytest.fixture(
#     scope="function",
#     params=(
#         CustomEventLoopPolicy(),
#         CustomEventLoopPolicy(),
#     ),
# )
# def event_loop_policy(request):
#     return request.param


# ORIG: https://gist.github.com/iedmrc/2fbddeb8ca8df25356d8acc3d297e955
@pytest.mark.asyncio(scope="module")
@pytest.mark.unittest
@pytest.mark.integration
class TestUtilsAsync:
    # SOURCE: https://pytest-asyncio.readthedocs.io/en/latest/how-to-guides/run_class_tests_in_same_loop.html
    loop: asyncio.AbstractEventLoop
    # @pytest.mark.asyncio
    # @pytest_asyncio.fixture
    # async def test_async_(self, monkeypatch: MonkeyPatch) -> None:

    async def test_remember_loop(self):
        TestUtilsAsync.loop = asyncio.get_running_loop()

    async def test_to_async(self):
        loop = asyncio.get_running_loop()

        @async_.to_async
        def time_sleep():
            import time

            time.sleep(1)
            return True

        import time

        # take the elapsed time
        start = time.time()
        res = await time_sleep()
        end = time.time()
        elapsed = end - start
        assert res == True

    # async def test_turn_sync_to_async(self):
    #     loop = asyncio.get_running_loop()
    #     @async_.force_async
    #     def test():
    #         import time

    #         time.sleep(1)
    #         return True

    #     @async_.force_sync
    #     async def main(loop):
    #         # import asyncio
    #         # if it were to execute sequentially, it would take 10 seconds, in this case we expect to see only 1 second
    #         futures = list(map(lambda x: test(), range(10)))
    #         return await loop.gather(*futures)

    #     import time

    #     # take the elapsed time
    #     start = time.time()
    #     res = main(loop)
    #     end = time.time()
    #     elapsed = end - start
    #     assert len(res) == 10
    #     assert elapsed < 1.2  # a little more than 1 second is normal

    # async def test_turn_async_to_sync(self):
    #     loop = TestUtilsAsync.loop
    #     @async_.force_sync
    #     async def test():
    #         # import asyncio

    #         await asyncio.sleep(0.1)
    #         return 1 + 2

    #     assert test() == 3

    # # @pytest.mark.asyncio
    # def test_turn_sync_to_sync(self):
    #     @async_.force_sync
    #     def test():
    #         return 1 + 2

    #     assert test() == 3
