"""test_goobredis"""

# pylint: disable=assigning-non-slot
# pylint: disable=consider-using-from-import
from __future__ import annotations

from typing import TYPE_CHECKING

from goob_ai.backend.cache.goobredis import get_driver

import pytest


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch
    from goob_ai.backend.cache.goobredis import GoobRedisClient

    from pytest_mock.plugin import MockerFixture

import asyncio


# pytestmark = pytest.mark.asyncio


# @pytest.mark.app_settings({"applications": ["guillotina", "guillotina.contrib.redis"]})
async def test_redis_ops():
    driver = await get_driver()
    assert driver.initialized
    assert driver.pool is not None

    await driver.set("test", "testdata", expire=10)
    result = await driver.get("test")
    assert result == b"testdata"

    await driver.set("test2", "testdata", expire=10)
    await driver.set("test3", "testdata", expire=10)
    await driver.set("test4", "testdata", expire=1)
    await driver.set("test5", "testdata", expire=20)
    await driver.set("test6", "testdata", expire=20)
    await driver.expire("test5", 1)

    await driver.delete("test")
    result = await driver.get("test")
    assert result is None

    result = await driver.keys_startswith("test")
    assert len(result) == 5

    await driver.delete_all(["test2", "test3"])
    result = await driver.get("test2")
    assert result is None

    await asyncio.sleep(2)
    result = await driver.get("test4")
    assert result is None

    result = await driver.get("test5")
    assert result is None

    await driver.flushall()
    result = await driver.get("test6")
    assert result is None

    result = await driver.info()
    await driver.finalize()
    assert driver.initialized is False


# @pytest.mark.app_settings({"applications": ["guillotina", "guillotina.contrib.redis"]})
async def test_redis_pubsub():
    driver = await get_driver()
    assert driver.initialized

    channel = await driver.subscribe("test::pubsub")
    assert channel is not None
    RESULTS = []

    async def receiver(callback):
        async for obj in callback:
            RESULTS.append(obj)
            break

    task = asyncio.ensure_future(receiver(channel))

    await driver.publish("test::pubsub", "dummydata")
    await asyncio.sleep(0.1)

    assert RESULTS
    assert RESULTS[0] == b"dummydata"

    task.cancel()
    await driver.finalize()
    assert driver.initialized is False
