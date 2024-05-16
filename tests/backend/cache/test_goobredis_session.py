"""test_goobredis_session"""

# pylint: disable=assigning-non-slot
# pylint: disable=consider-using-from-import
from __future__ import annotations

import asyncio

from typing import TYPE_CHECKING

from goob_ai.backend.cache import goobredis_session

import pytest


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch
    from goob_ai.backend.cache.goobredis import GoobRedisClient
    from goob_ai.backend.cache.goobredis_session import RedisSessionManagerUtility

    from pytest_mock.plugin import MockerFixture


async def test_redis_session_manager_utility():
    manager: RedisSessionManagerUtility = goobredis_session.RedisSessionManagerUtility()
    await manager.initialize()
    assert manager._prefix == "session"
    assert manager._ttl == 3660
    assert manager._initialized
    assert str(type(manager._driver)) == "<class 'goob_ai.backend.cache.goobredis.GoobRedisClient'>"
    test_session_uuid = await manager.new_session("test", "testdata")

    assert await manager.exist_session("test", test_session_uuid)
    sessions = await manager.list_sessions("test")
    assert len(sessions) >= 1
    await asyncio.sleep(1)

    test_session_uuid = await manager.refresh_session("test", test_session_uuid)
    assert test_session_uuid
    await asyncio.sleep(1)

    res = await manager.get_session("test", test_session_uuid)
    assert len(res) >= 1
    await asyncio.sleep(1)

    res = await manager.drop_session("test", test_session_uuid)
    await asyncio.sleep(1)

    # assert len(res) == 0

    # session:test:e07deaa2d4ca46d6ac05de8e43820b44
