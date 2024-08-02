"""test_goobredis"""

# pylint: disable=assigning-non-slot
# pylint: disable=consider-using-from-import
from __future__ import annotations

import asyncio

from typing import TYPE_CHECKING, Optional

import pytest_asyncio
import redis.asyncio as redis

from goob_ai.backend.cache.goobredis import get_driver
from loguru import logger as LOGGER
from redis.connection import Connection, parse_url
from redis.exceptions import RedisClusterException
from redis.retry import Retry

import pytest


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from goob_ai.backend.cache.goobredis import GoobRedisClient

    from pytest_mock.plugin import MockerFixture

##########################################################################


# # @pytest_asyncio.fixture(loop_scope="module")
# # async def current_loop():
# #     return asyncio.get_running_loop()

# # Creates a new asyncio event loop based on the current event loop policy. The new loop is available as the return value of this fixture for synchronous functions, or via asyncio.get_running_loop for asynchronous functions. The event loop is closed when the fixture scope ends. The fixture scope defaults to function scope.
# @pytest.mark.asyncio(loop_scope="module")
# async def _get_client(
#     event_loop, request, capsys, caplog, cls=cls=redis.Redis,single_connection_client: bool=True, flushdb: bool=False, from_url: Optional[str]=None, **kwargs
# ):
#     """
#     Helper for fixtures or tests that need a Redis client

#     Uses the "--redis-url" command line argument for connection info. Unlike
#     ConnectionPool.from_url, keyword arguments to this function override
#     values specified in the URL.
#     """

#     loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

#     # pool = redis.ConnectionPool(**url_options)
#     pool: GoobRedisClient = await get_driver()
#     pool.initialize(loop)
#     client = cls(connection_pool=pool)
#     # if not cluster_mode:
#     #     url_options = parse_url(redis_url)
#     #     url_options.update(kwargs)
#     #     pool = redis.ConnectionPool(**url_options)
#     #     client = cls(connection_pool=pool)
#     # else:
#     #     client = redis.RedisCluster.from_url(redis_url, **kwargs)
#     #     single_connection_client = False
#     if single_connection_client:
#         client = client.client()
#     if request:

#         def teardown():
#             # if not cluster_mode:
#             #     if flushdb:
#             #         try:
#             #             client.flushdb()
#             #         except redis.ConnectionError:
#             #             # handle cases where a test disconnected a client
#             #             # just manually retry the flushdb
#             #             client.flushdb()
#             #     client.close()
#             #     client.connection_pool.disconnect()
#             # else:
#             cluster_teardown(client, flushdb)

#         request.addfinalizer(teardown)
#     return client


def cluster_teardown(client, flushdb):
    if flushdb:
        try:
            client.flushdb(target_nodes="primaries")
        except redis.ConnectionError:
            # handle cases where a test disconnected a client
            # just manually retry the flushdb
            client.flushdb(target_nodes="primaries")
    client.close()
    client.disconnect_connection_pools()


async def aio_cluster_teardown(client, flushdb):
    if flushdb:
        try:
            client.flushdb(target_nodes="primaries")
        except redis.ConnectionError:
            # handle cases where a test disconnected a client
            # just manually retry the flushdb
            client.flushdb(target_nodes="primaries")
    client.close()
    client.disconnect_connection_pools()


# @pytest.fixture(name="create_redis")
@pytest_asyncio.fixture(name="create_redis")
async def create_redis(
    event_loop: asyncio.AbstractEventLoop, request: type[FixtureRequest], caplog: LogCaptureFixture
):  # -> Generator[Callable[..., Coroutine[Any, Any, GoobRedisClient]], Any, None]:
    """Wrapper around redis.create_redis."""
    # (single_connection,) = request.param

    teardown_clients = []
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

    async def client_factory(
        **kwargs,
    ) -> GoobRedisClient:
        # SOURCE: https://github.com/redis/redis-py/blob/fb74aa2806100f4026e290dab0cb7164262ff142/tests/test_asyncio/conftest.py#L41
        client: GoobRedisClient = await get_driver()
        # client.single_connection_client = True
        # client.initialize(loop)

        # client = client.client()
        # await client.initialize()

        async def teardown():
            try:
                await client.aclose()
                # await client.pool.connection_pool.disconnect()
                # await client._pool.aclose()
                # await client.pool.disconnect()
            except redis.ConnectionError:
                await client.aclose()
                # await client.pool.connection_pool.disconnect()
                pass

        teardown_clients.append(teardown)
        return client

    yield client_factory

    for teardown in teardown_clients:
        await teardown()


@pytest_asyncio.fixture(name="r")
async def r(create_redis):
    return await create_redis()


@pytest_asyncio.fixture(name="decoded_r")
async def decoded_r(create_redis):
    return await create_redis(decode_responses=True)


##########################################################################


@pytest.mark.asyncio()
# @pytest.mark.app_settings({"applications": ["guillotina", "guillotina.contrib.redis"]})
async def test_redis_ops(caplog: LogCaptureFixture, create_redis, **kwargs):
    # driver: GoobRedisClient = await get_driver()
    driver = await create_redis(**kwargs)
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

    # wait for logging to finish runnnig
    await LOGGER.complete()

    caplog.clear()


# @pytest.mark.asyncio()
# # @pytest.mark.app_settings({"applications": ["guillotina", "guillotina.contrib.redis"]})
# async def test_redis_pubsub(caplog: LogCaptureFixture):
#     driver = await get_driver()
#     assert driver.initialized

#     channel = await driver.subscribe("test::pubsub")
#     assert channel is not None
#     RESULTS = []

#     async def receiver(callback):
#         async for obj in callback:
#             RESULTS.append(obj)
#             break

#     task = asyncio.ensure_future(receiver(channel))

#     await driver.publish("test::pubsub", "dummydata")
#     await asyncio.sleep(0.1)

#     assert RESULTS
#     assert RESULTS[0] == b"dummydata"

#     task.cancel()
#     await driver.finalize()
#     assert driver.initialized is False
