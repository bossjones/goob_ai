# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
# SOURCE: https://github.com/dataelement/bisheng/blob/main/src/backend/bisheng/cache/redis.py
# SOURCE: https://github.com/plone/guillotina/tree/4bf16089dab1065fa39211b45d1a5101d6ea5e84

from __future__ import annotations

import asyncio
import pickle

from typing import Any, Dict, List, Optional

import backoff
import redis
import redis.asyncio

from loguru import logger as LOGGER
from redis.asyncio import ConnectionPool, Redis
from redis.asyncio.client import PubSub
from redis.backoff import ExponentialBackoff
from redis.cluster import ClusterNode
from redis.exceptions import ConnectionError
from redis.retry import Retry
from redis.sentinel import Sentinel

from goob_ai import metrics
from goob_ai.aio_settings import AioSettings, aiosettings


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
    pass


class GoobRedisClient:
    def __init__(self, url, max_connections=10):
        self._pool: Redis | None = None  # pyright: ignore[reportAttributeAccessIssue]
        self._pubsub = None
        self._loop = None
        self._receivers = {}
        self._pubsub_subscriptor = None
        self._conn = None
        self.initialized = False
        self.init_lock = asyncio.Lock()
        self._max_connections = max_connections

        # self.pool: ConnectionPool = ConnectionPool.from_url(str(aiosettings.redis_url), max_connections=max_connections)

        # self.connection: Redis = redis.asyncio.Redis(connection_pool=self.pool)

    async def initialize(self, loop):
        self._loop = loop
        async with self.init_lock:
            if self.initialized is False:
                while True:
                    try:
                        await self._connect()
                        with watch("acquire_conn"):
                            assert await self._pool.ping() is True
                        self.initialized = True
                        break
                    except Exception:  # pragma: no cover
                        LOGGER.error("Error initializing pubsub", exc_info=True)

    @backoff.on_exception(backoff.expo, (OSError,), max_time=30, max_tries=4)
    async def _connect(self):
        # settings = app_settings["redis"]
        self._conn_pool: ConnectionPool = redis.asyncio.ConnectionPool.from_url(
            str(aiosettings.redis_url), max_connections=self._max_connections
        )
        self._pool: Redis = redis.asyncio.Redis(connection_pool=self._conn_pool)
        self._pubsub_channels: Dict[str, PubSub] = {}

    async def finalize(self):
        await self._conn_pool.disconnect()
        self.initialized = False

    @property
    def pool(self):
        return self._pool

    async def info(self):
        if self._pool is None:
            raise NoRedisConfigured()
        return await self._pool.info("get")

    # VALUE API

    async def set(self, key: str, data: str, *, expire: Optional[int] = None):
        if self._pool is None:
            raise NoRedisConfigured()
        kwargs = {}
        if expire is not None:
            kwargs["ex"] = expire
        with watch("set"):
            ok = await self._pool.set(key, data, **kwargs)  # mypy: disable-error-code="arg-type"
        assert ok is True, ok

    async def get(self, key: str) -> str | Any | None:
        if self._pool is None:
            raise NoRedisConfigured()
        with watch("get") as w:
            val = await self._pool.get(key)
            if not val:
                w.labels["type"] = "get_miss"
            return val

    async def delete(self, key: str):
        if self._pool is None:
            raise NoRedisConfigured()
        with watch("delete"):
            await self._pool.delete(key)

    async def expire(self, key: str, expire: int):
        if self._pool is None:
            raise NoRedisConfigured()
        await self._pool.expire(key, expire)

    async def keys_startswith(self, key: str):
        if self._pool is None:
            raise NoRedisConfigured()
        return await self._pool.keys(f"{key}*")

    async def delete_all(self, keys: List[str]):
        if self._pool is None:
            raise NoRedisConfigured()
        for key in keys:
            try:
                with watch("delete_many"):
                    await self._pool.delete(key)
                LOGGER.debug("Deleted cache keys {}".format(keys))
            except Exception:
                LOGGER.warning("Error deleting cache keys {}".format(keys), exc_info=True)

    async def flushall(self, *, async_op: bool = False):
        if self._pool is None:
            raise NoRedisConfigured()
        with watch("flush"):
            await self._pool.flushdb(asynchronous=async_op)

    # PUBSUB API

    async def publish(self, channel_name: str, data: str):
        if self._pool is None:
            raise NoRedisConfigured()

        with watch("publish"):
            await self._pool.publish(channel_name, data)

    async def unsubscribe(self, channel_name: str):
        if self._pool is None:
            raise NoRedisConfigured()

        p = self._pubsub_channels.pop(channel_name, None)
        if p:
            try:
                await p.unsubscribe(channel_name)
            except ConnectionError:
                LOGGER.error(f"Error unsubscribing channel {channel_name}", exc_info=True)
            finally:
                await p.__aexit__(None, None, None)

    async def subscribe(self, channel_name: str):
        if self._pool is None:
            raise NoRedisConfigured()

        p: PubSub = await self._pool.pubsub().__aenter__()
        self._pubsub_channels[channel_name] = p
        await p.subscribe(channel_name)
        return self._listener(p)

    async def _listener(self, p: PubSub):
        while True:
            message = await p.get_message(ignore_subscribe_messages=True, timeout=1)
            if message is not None:
                yield message["data"]


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

_DRIVER = GoobRedisClient(str(aiosettings.redis_url))


async def get_driver() -> GoobRedisClient:
    global _driver
    if _DRIVER is None:
        raise Exception("Not added goob_ai.contrib.redis on applications")
    else:
        if _DRIVER.initialized is False:
            loop = asyncio.get_event_loop()
            await _DRIVER.initialize(loop)
        return _DRIVER


# redis_client = await get_driver()
