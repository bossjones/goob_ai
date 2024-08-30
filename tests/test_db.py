from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Optional, TypeVar

from goob_ai.aio_settings import aiosettings
from goob_ai.db import (
    Base,
    RedisValueDTO,
    async_connection_db,
    create_async_engine_db,
    get_db,
    get_redis_conn_pool,
    get_redis_value,
    init_worker_redis,
    set_redis_value,
)
from loguru import logger as LOGGER
from redis.asyncio import ConnectionPool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

import pytest


if TYPE_CHECKING:
    from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock

    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture()
def db_session():
    """
    Fixture to provide a database session for testing.

    Yields:
        Session: A SQLAlchemy session for testing.
    """
    yield from get_db()


def test_get_db(db_session):
    """
    Test the get_db function.

    Args:
        db_session: The database session fixture.
    """
    assert db_session is not None
    result = db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1


@pytest.mark.asyncio()
async def test_create_async_engine_db():
    """Test the create_async_engine_db function."""
    engine = await create_async_engine_db()
    assert isinstance(engine, AsyncEngine)
    await engine.dispose()


@pytest.mark.asyncio()
async def test_async_connection_db():
    """Test the async_connection_db function."""
    engine = await create_async_engine_db()
    async_session = await async_connection_db(engine, expire_on_commit=False)
    assert callable(async_session)
    session: AsyncSession = async_session()
    assert isinstance(session, AsyncSession)
    await session.close()
    await engine.dispose()


def test_init_worker_redis():
    """Test the init_worker_redis function."""
    redis_pool = init_worker_redis()
    assert isinstance(redis_pool, ConnectionPool)
    redis_pool.disconnect()


def test_get_redis_conn_pool():
    """Test the get_redis_conn_pool function."""
    redis_pool = get_redis_conn_pool()
    assert isinstance(redis_pool, ConnectionPool)
    redis_pool.disconnect()


@pytest.mark.asyncio()
async def test_get_redis_value():
    """Test the get_redis_value function."""
    redis_pool = get_redis_conn_pool()
    key = "test_key"
    value = "test_value"

    # Set a test value
    await set_redis_value(RedisValueDTO(key=key, value=value), redis_pool)

    # Get the value
    result = await get_redis_value(key, redis_pool)
    assert isinstance(result, RedisValueDTO)
    assert result.key == key
    assert result.value == value

    redis_pool.disconnect()


@pytest.mark.asyncio()
async def test_set_redis_value():
    """Test the set_redis_value function."""
    redis_pool = get_redis_conn_pool()
    key = "test_set_key"
    value = "test_set_value"

    # Set the value
    await set_redis_value(RedisValueDTO(key=key, value=value), redis_pool)

    # Verify the value was set
    result = await get_redis_value(key, redis_pool)
    assert result.value == value

    redis_pool.disconnect()


@pytest.mark.asyncio()
async def test_set_redis_value_none():
    """Test the set_redis_value function with None value."""
    redis_pool = get_redis_conn_pool()
    key = "test_none_key"

    # Set the value to None
    await set_redis_value(RedisValueDTO(key=key, value=None), redis_pool)

    # Verify the value was not set
    result = await get_redis_value(key, redis_pool)
    assert result.value is None

    redis_pool.disconnect()
