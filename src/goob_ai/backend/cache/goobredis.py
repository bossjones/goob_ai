# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
# SOURCE: https://github.com/dataelement/bisheng/blob/main/src/backend/bisheng/cache/redis.py
# SOURCE: https://github.com/plone/guillotina/tree/4bf16089dab1065fa39211b45d1a5101d6ea5e84
# FIXME: Look at this and make sure you incorporate it into your code. https://github.com/CL-lau/SQL-GPT/blob/47c2e9434855ce699da556bbbc043cbee2ae3540/memory/redisHelper.py

from __future__ import annotations

import asyncio
import pickle
import sys
import traceback

from collections.abc import AsyncGenerator
from typing import Any, Callable, Dict, List, Optional, Union

import backoff
import redis
import redis.asyncio

from loguru import logger as LOGGER
from redis.asyncio import Connection, Redis
from redis.asyncio.client import PubSub
from redis.asyncio.connection import ConnectionPool as AsyncConnectionPool
from redis.backoff import ExponentialBackoff
from redis.cluster import ClusterNode
from redis.exceptions import ConnectionError
from redis.retry import Retry
from redis.sentinel import Sentinel

from goob_ai import metrics
from goob_ai.aio_settings import AioSettings, aiosettings


# from langchain.cache import RedisCache
# from redis.asyncio.connection import (
#     BlockingConnectionPool,
#     Connection,
#     ConnectionPool,
#     SSLConnection,
#     UnixDomainSocketConnection,
# )


try:
    import prometheus_client

    REDIS_OPS = prometheus_client.Counter(
        "goob_ai_cache_redis_ops_total",
        "Total count of ops by type of operation and the error if there was.",
        labelnames=["type", "error"],
    )
    REDIS_OPS_PROCESSING_TIME = prometheus_client.Histogram(
        "goob_ai_cache_redis_ops_processing_time_seconds",
        "Histogram of operations processing time by type (in seconds)",
        labelnames=["type"],
    )

    class watch(metrics.watch):
        def __init__(self, operation: str):
            super().__init__(
                counter=REDIS_OPS,
                histogram=REDIS_OPS_PROCESSING_TIME,
                labels={"type": operation},
            )

except ImportError:
    watch = metrics.watch  # type: ignore


class NoRedisConfigured(Exception):
    """Exception raised when no Redis is configured."""


class GoobRedisClient:
    """
    A Redis client class for Goob AI.

    Args:
        url: The URL of the Redis server.
        max_connections: The maximum number of connections to the Redis server.
    """

    def __init__(self, url: str, max_connections: int = 10):
        self._url: str = url
        self._pool: Optional[Redis] = None
        self._pubsub: Optional[PubSub] = None
        self._loop = None
        self._receivers: dict[str, Any] = {}
        self._pubsub_subscriptor = None
        self._conn: Optional[Connection] = None
        self.connection: Optional[Connection] = None
        self.initialized = False
        self.init_lock = asyncio.Lock()
        self._max_connections: int = max_connections
        self.auto_close_connection_pool: Optional[bool] = None
        self._client: Optional[Redis] = None

    async def initialize(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Initialize the Redis client.

        Args:
            loop: The event loop to use.
        """
        if self._loop is None:
            self._loop = loop
        async with self.init_lock:
            if not self.initialized:
                while True:
                    try:
                        await self._connect()
                        with watch("acquire_conn"):
                            assert await self._pool.ping() is True
                        self.initialized = True
                        break
                    except Exception as ex:  # pragma: no cover
                        LOGGER.error("Error initializing pubsub", exc_info=True)
                        print(f"{ex}")
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        print(f"Error Class: {ex.__class__}")
                        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
                        print(output)
                        print(f"exc_type: {exc_type}")
                        print(f"exc_value: {exc_value}")
                        traceback.print_tb(exc_traceback)
                        # if aiosettings.dev_mode:
                        #     bpdb.pm()

    @backoff.on_exception(backoff.expo, (OSError,), max_time=30, max_tries=4)
    async def _connect(self) -> None:
        """Connect to the Redis server."""

        # If you create a custom `ConnectionPool` to be used by a single `Redis` instance, use the `Redis.from_pool` class method. The Redis client will take ownership of the connection pool. This will cause the pool to be disconnected along with the Redis instance. Disconnecting the connection pool simply disconnects all connections hosted in the pool.
        self._conn_pool: Union[redis.asyncio.AsyncConnectionPool, redis.asyncio.ConnectionPool] = (
            redis.asyncio.ConnectionPool.from_url(str(aiosettings.redis_url), max_connections=self._max_connections)
        )
        # , single_connection_client=True

        # Create redis client now using the connection pool
        self._client: Redis = redis.asyncio.Redis.from_pool(connection_pool=self._conn_pool)

        # NOTE: We might need to override self._client
        # self._client.client()

        self._conn: Connection = await self._conn_pool.get_connection(":")

        # If you create a custom ConnectionPool to be used by a single Redis instance, use the Redis.from_pool class method. The Redis client will take ownership of the connection pool. This will cause the pool to be disconnected along with the Redis instance. Disconnecting the connection pool simply disconnects all connections hosted in the pool.
        self._pool: redis.asyncio.Redis = redis.asyncio.Redis.from_pool(self._conn_pool)

        # What delimiter do you use for your #Redis keys? Select Option 1 for colons(:), Option 2 for underscores(_), Option 3 for hyphens(-), or Option 4 if you use some other character as a delimiter or don't use a delimiter at all.

        # self.connection: Connection = redis.asyncio.StrictRedis(connection_pool=self._pool)
        # self._conn: Connection = redis.asyncio.StrictRedis(connection_pool=self._pool)

        self._pubsub_channels: dict[str, PubSub] = {}
        self.auto_close_connection_pool: bool = self._pool.auto_close_connection_pool

    async def finalize(self) -> None:
        """Finalize the Redis client."""
        await self._conn_pool.disconnect()
        self.initialized = False

    @property
    def pool(self) -> Optional[Redis]:
        """Get the Redis connection pool."""
        return self._pool

    async def info(self) -> dict[str, Any]:
        """
        Get information about the Redis server.

        Returns:
            A dictionary containing information about the Redis server.

        Raises:
            NoRedisConfigured: If no Redis is configured.
        """
        if self._pool is None:
            raise NoRedisConfigured()
        return await self._pool.info("get")

    async def set(self, key: str, data: str, *, expire: Optional[int] = None) -> None:
        """
        Set a key-value pair in Redis.

        Args:
            key: The key to set.
            data: The value to set.
            expire: The expiration time in seconds.

        Raises:
            NoRedisConfigured: If no Redis is configured.
        """
        if self._pool is None:
            raise NoRedisConfigured()
        kwargs = {}
        if expire is not None:
            kwargs["ex"] = expire
        with watch("set"):
            ok = await self._pool.set(key, data, **kwargs)  # type: ignore
        assert ok is True, ok

    async def get(self, key: str) -> Optional[str]:
        """
        Get the value of a key from Redis.

        Args:
            key: The key to get.

        Returns:
            The value of the key, or None if the key does not exist.

        Raises:
            NoRedisConfigured: If no Redis is configured.
        """
        if self._pool is None:
            raise NoRedisConfigured()
        with watch("get") as w:
            val = await self._pool.get(key)
            if not val:
                w.labels["type"] = "get_miss"
            return val

    async def delete(self, key: str) -> None:
        """
        Delete a key from Redis.

        Args:
            key: The key to delete.

        Raises:
            NoRedisConfigured: If no Redis is configured.
        """
        if self._pool is None:
            raise NoRedisConfigured()
        with watch("delete"):
            await self._pool.delete(key)

    async def expire(self, key: str, expire: int) -> None:
        """
        Set the expiration time for a key in Redis.

        Args:
            key: The key to set the expiration for.
            expire: The expiration time in seconds.

        Raises:
            NoRedisConfigured: If no Redis is configured.
        """
        if self._pool is None:
            raise NoRedisConfigured()
        await self._pool.expire(key, expire)

    async def keys_startswith(self, key: str) -> list[str]:
        """
        Get all keys that start with a given prefix.

        Args:
            key: The prefix to search for.

        Returns:
            A list of keys that start with the given prefix.

        Raises:
            NoRedisConfigured: If no Redis is configured.
        """
        if self._pool is None:
            raise NoRedisConfigured()
        return await self._pool.keys(f"{key}*")

    async def delete_all(self, keys: list[str]) -> None:
        """
        Delete multiple keys from Redis.

        Args:
            keys: The list of keys to delete.

        Raises:
            NoRedisConfigured: If no Redis is configured.
        """
        if self._pool is None:
            raise NoRedisConfigured()
        for key in keys:
            try:
                with watch("delete_many"):
                    await self._pool.delete(key)
                LOGGER.debug(f"Deleted cache keys {keys}")
            except Exception:
                LOGGER.warning(f"Error deleting cache keys {keys}", exc_info=True)

    async def flushall(self, *, async_op: bool = False) -> None:
        """
        Flush all keys from Redis.

        Args:
            async_op: Whether to perform the operation asynchronously.

        Raises:
            NoRedisConfigured: If no Redis is configured.
        """
        if self._pool is None:
            raise NoRedisConfigured()
        with watch("flush"):
            await self._pool.flushdb(asynchronous=async_op)

    async def publish(self, channel_name: str, data: str) -> None:
        """
        Publish a message to a Redis channel.

        Args:
            channel_name: The name of the channel to publish to.
            data: The message to publish.

        Raises:
            NoRedisConfigured: If no Redis is configured.
        """
        if self._pool is None:
            raise NoRedisConfigured()

        with watch("publish"):
            await self._pool.publish(channel_name, data)

    async def unsubscribe(self, channel_name: str) -> None:
        """
        Unsubscribe from a Redis channel.

        Args:
            channel_name: The name of the channel to unsubscribe from.

        Raises:
            NoRedisConfigured: If no Redis is configured.
        """
        if self._pool is None:
            raise NoRedisConfigured()

        if p := self._pubsub_channels.pop(channel_name, None):
            try:
                await p.unsubscribe(channel_name)
            except ConnectionError:
                LOGGER.error(f"Error unsubscribing channel {channel_name}", exc_info=True)
            finally:
                await p.__aexit__(None, None, None)

    async def subscribe(self, channel_name: str) -> AsyncGenerator[Any, None]:
        """
        Subscribe to a Redis channel.

        Args:
            channel_name: The name of the channel to subscribe to.

        Yields:
            Messages received from the subscribed channel.

        Raises:
            NoRedisConfigured: If no Redis is configured.
        """
        if self._pool is None:
            raise NoRedisConfigured()

        p: PubSub = await self._pool.pubsub().__aenter__()
        self._pubsub_channels[channel_name] = p
        await p.subscribe(channel_name)
        return self._listener(p)

    async def _listener(self, p: PubSub) -> AsyncGenerator[Any, None]:
        """
        Listen for messages on a Redis channel.

        Args:
            p: The PubSub object to listen on.

        Yields:
            Messages received from the subscribed channel.
        """
        while True:
            message = await p.get_message(ignore_subscribe_messages=True, timeout=1)
            if message is not None:
                yield message["data"]

    async def aclose(self, close_connection_pool: Optional[bool] = None) -> None:
        """
        Closes Redis client connection

        :param close_connection_pool: decides whether to close the connection pool used
        by this Redis client, overriding Redis.auto_close_connection_pool. By default,
        let Redis.auto_close_connection_pool decide whether to close the connection
        pool.
        """
        conn: Connection | None = self._conn
        if conn:
            self._conn = None
            await self._pool.connection_pool.release(conn)
        if close_connection_pool or (close_connection_pool is None and self.auto_close_connection_pool):
            await self._pool.connection_pool.disconnect()


_DRIVER: Optional[GoobRedisClient] = GoobRedisClient(str(aiosettings.redis_url))


async def get_driver() -> GoobRedisClient:
    """
    Get the Redis client driver.

    Returns:
        The Redis client driver.

    Raises:
        Exception: If the Redis client is not added to the applications.
    """
    global _DRIVER
    if _DRIVER is None:
        raise Exception("Not added goob_ai.contrib.redis on applications")
    if not _DRIVER.initialized:
        loop = asyncio.get_event_loop()
        await _DRIVER.initialize(loop)
    return _DRIVER


# redis_client = await get_driver()

# async def aclose(self, close_connection_pool: Optional[bool] = None) -> None:
#     """
#     Closes Redis client connection

#     :param close_connection_pool: decides whether to close the connection pool used
#     by this Redis client, overriding Redis.auto_close_connection_pool. By default,
#     let Redis.auto_close_connection_pool decide whether to close the connection
#     pool.
#     """
#     conn = self.connection
#     if conn:
#         self.connection = None
#         await self.connection_pool.release(conn)
#     if close_connection_pool or (
#         close_connection_pool is None and self.auto_close_connection_pool
#     ):
#         await self.connection_pool.disconnect()


#     def set(self, key, value, expiration=3600):
#         try:
#             if pickled := pickle.dumps(value):
#                 self.cluster_nodes(key)
#                 result = self.connection.setex(key, expiration, pickled)
#                 if not result:
#                     raise ValueError('RedisCache could not set the value.')
#             else:
#                 LOGGER.error('pickle error, value={}', value)
#         except TypeError as exc:
#             raise TypeError('RedisCache only accepts values that can be pickled. ') from exc
#         finally:
#             self.close()

#     def setNx(self, key, value, expiration=3600):
#         try:
#             if pickled := pickle.dumps(value):
#                 self.cluster_nodes(key)
#                 result = self.connection.setnx(key, pickled)
#                 self.connection.expire(key, expiration)
#                 if not result:
#                     return False
#                 return True
#         except TypeError as exc:
#             raise TypeError('RedisCache only accepts values that can be pickled. ') from exc
#         finally:
#             self.close()

#     def hsetkey(self, name, key, value, expiration=3600):
#         try:
#             self.cluster_nodes(key)
#             r = self.connection.hset(name, key, value)
#             if expiration:
#                 self.connection.expire(name, expiration)
#             return r
#         finally:
#             self.close()

#     def hset(self, name, map: dict, expiration=3600):
#         try:
#             self.cluster_nodes(name)
#             r = self.connection.hset(name, mapping=map)
#             if expiration:
#                 self.connection.expire(name, expiration)
#             return r
#         finally:
#             self.close()

#     def hget(self, name, key):
#         try:
#             self.cluster_nodes(name)
#             return self.connection.hget(name, key)
#         finally:
#             self.close()

#     def get(self, key):
#         try:
#             self.cluster_nodes(key)
#             value = self.connection.get(key)
#             return pickle.loads(value) if value else None
#         finally:
#             self.close()

#     def delete(self, key):
#         try:
#             self.cluster_nodes(key)
#             return self.connection.delete(key)
#         finally:
#             self.close()

#     def exists(self, key):
#         try:
#             self.cluster_nodes(key)
#             return self.connection.exists(key)
#         finally:
#             self.close()

#     def close(self):
#         self.connection.close()

#     def __contains__(self, key):
#         """Check if the key is in the cache."""
#         self.cluster_nodes(key)
#         return False if key is None else self.connection.exists(key)

#     def __getitem__(self, key):
#         """Retrieve an item from the cache using the square bracket notation."""
#         self.cluster_nodes(key)
#         return self.connection.get(key)

#     def __setitem__(self, key, value):
#         """Add an item to the cache using the square bracket notation."""
#         self.cluster_nodes(key)
#         self.connection.set(key, value)

#     def __delitem__(self, key):
#         """Remove an item from the cache using the square bracket notation."""
#         self.cluster_nodes(key)
#         self.connection.delete(key)

#     def cluster_nodes(self, key):
#         if isinstance(self.connection,
#                       RedisCluster) and self.connection.get_default_node() is None:
#             target = self.connection.get_node_from_key(key)
#             self.connection.set_default_node(target)


# # Example usage
# redis_client = RedisClient(f"{aiosettings.redis_url}")
