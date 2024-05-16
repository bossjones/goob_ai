# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
# pylint: disable=simplifiable-if-statement
# SOURCE: https://github.com/dataelement/bisheng/blob/main/src/backend/bisheng/cache/redis.py
# SOURCE: https://github.com/plone/guillotina/tree/4bf16089dab1065fa39211b45d1a5101d6ea5e84

from __future__ import annotations

import asyncio
import logging
import pickle
import uuid

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


class RedisSessionManagerUtility:
    def __init__(self):
        self._ttl = 3660
        self._prefix = "session"
        self._driver = None
        self._initialized = False

    async def initialize(self, app=None):
        from goob_ai.backend.cache import goobredis

        loop = asyncio.get_event_loop()
        self._driver = await goobredis.get_driver()
        await self._driver.initialize(loop)
        self._initialized = True

    async def finalize(self):
        self._initialized = False

    async def new_session(self, ident: str, data: str = "") -> str:
        session = uuid.uuid4().hex
        session_key = f"{self._prefix}:{ident}:{session}"
        await self._driver.set(session_key, data, expire=self._ttl)
        return session

    async def exist_session(self, ident: str | None, session: str | None) -> bool:
        if session is None:
            return False
        if ident is None:
            return False
        session_key = f"{self._prefix}:{ident}:{session}"
        value = await self._driver.get(session_key)
        if value is not None:
            return True
        else:
            return False

    async def drop_session(self, ident: str, session: str):
        session_key = f"{self._prefix}:{ident}:{session}"
        value = await self._driver.get(session_key)
        if value is not None:
            await self._driver.delete(session_key)
        else:
            raise KeyError("Invalid session")

    async def refresh_session(self, ident: str, session: str):
        session_key = f"{self._prefix}:{ident}:{session}"
        value = await self._driver.get(session_key)
        if value is not None:
            await self._driver.expire(session_key, self._ttl)
            return session
        else:
            raise KeyError("Invalid session")

    async def list_sessions(self, ident: str | None):
        if ident is None:
            return []
        session_key = f"{self._prefix}:{ident}"
        value = await self._driver.keys_startswith(session_key)
        return [x.split(b":")[2].decode("utf-8") for x in value]

    async def get_session(self, ident: str | None, session: str | None):
        if ident is None:
            return []
        session_key = f"{self._prefix}:{ident}:{session}"
        value = await self._driver.get(session_key)
        return value.decode("utf-8")  # pyright: ignore[reportAttributeAccessIssue]
