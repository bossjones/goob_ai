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
from goob_ai.backend.cache.goobredis import GoobRedisClient


class RedisSessionManagerUtility:
    """Utility class for managing sessions using Redis."""

    def __init__(self):
        self._ttl = 3660
        self._prefix = "session"
        self._driver: Optional[GoobRedisClient] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Redis session manager."""
        from goob_ai.backend.cache import goobredis

        loop = asyncio.get_event_loop()
        self._driver = await goobredis.get_driver()
        await self._driver.initialize(loop)
        self._initialized = True

    async def finalize(self) -> None:
        """Finalize the Redis session manager."""
        self._initialized = False

    async def new_session(self, ident: str, data: str = "") -> str:
        """
        Create a new session.

        Args:
            ident: The identifier for the session.
            data: The data to store in the session.

        Returns:
            The session ID.
        """
        session = uuid.uuid4().hex
        session_key = f"{self._prefix}:{ident}:{session}"
        LOGGER.debug(f"Creating new session: session_key={session_key}")
        await self._driver.set(session_key, data, expire=self._ttl)
        return session

    async def exist_session(self, ident: Optional[str], session: Optional[str]) -> bool:
        """
        Check if a session exists.

        Args:
            ident: The identifier for the session.
            session: The session ID.

        Returns:
            True if the session exists, False otherwise.
        """
        if session is None or ident is None:
            return False
        session_key = f"{self._prefix}:{ident}:{session}"
        LOGGER.debug(f"Existing session: session_key={session_key}")
        value = await self._driver.get(session_key)
        return value is not None

    async def drop_session(self, ident: str, session: str) -> None:
        """
        Drop a session.

        Args:
            ident: The identifier for the session.
            session: The session ID.

        Raises:
            KeyError: If the session is invalid.
        """
        session_key = f"{self._prefix}:{ident}:{session}"
        LOGGER.debug(f"Drop session: session_key={session_key}")
        value = await self._driver.get(session_key)
        if value is not None:
            await self._driver.delete(session_key)
        else:
            raise KeyError("Invalid session")

    async def refresh_session(self, ident: str, session: str) -> str:
        """
        Refresh a session.

        Args:
            ident: The identifier for the session.
            session: The session ID.

        Returns:
            The refreshed session ID.

        Raises:
            KeyError: If the session is invalid.
        """
        session_key = f"{self._prefix}:{ident}:{session}"
        LOGGER.debug(f"Refresh session: session_key={session_key}")
        value = await self._driver.get(session_key)
        if value is not None:
            await self._driver.expire(session_key, self._ttl)
            LOGGER.debug(f"Refreshed: session_key={session_key}")
            LOGGER.debug(f"Refreshed: value={value}")
            return session
        else:
            raise KeyError("Invalid session")

    async def list_sessions(self, ident: Optional[str]) -> list[str]:
        """
        List all sessions for a given identifier.

        Args:
            ident: The identifier for the sessions.

        Returns:
            A list of session IDs.
        """
        if ident is None:
            return []
        session_key = f"{self._prefix}:{ident}"
        LOGGER.debug(f"List session: session_key={session_key}")
        value = await self._driver.keys_startswith(session_key)
        LOGGER.debug(f"List session: value={value}")
        return [x.split(b":")[2].decode("utf-8") for x in value]

    async def get_session(self, ident: Optional[str], session: Optional[str]) -> str:
        """
        Get the data for a session.

        Args:
            ident: The identifier for the session.
            session: The session ID.

        Returns:
            The session data.
        """
        if ident is None:
            return ""
        session_key = f"{self._prefix}:{ident}:{session}"
        LOGGER.debug(f"Get session: session_key={session_key}")
        value = await self._driver.get(session_key)
        return value.decode("utf-8") if value else ""
