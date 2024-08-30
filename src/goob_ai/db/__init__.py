"""goob_ai.db module."""

from __future__ import annotations

import asyncio

from collections.abc import AsyncGenerator, Generator
from typing import Any, Optional

from langchain.pydantic_v1 import BaseModel
from redis.asyncio import ConnectionPool, Redis
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from goob_ai.aio_settings import aiosettings


# Creating the engine
engine = create_engine(aiosettings.postgres_url)

# Creating the session factory
SessionLocal: sessionmaker[Session] = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base: Any = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Get a database session.

    Yields:
        Session: The database session.
    """
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()


async def create_async_engine_db(
    url: str = aiosettings.postgres_url,
    echo: bool = True,
) -> AsyncEngine:
    """
    Create an async database engine.

    Args:
        url (str): The database URL.
        echo (bool): Whether to echo SQL statements.

    Returns:
        AsyncEngine: The async database engine.
    """
    return create_async_engine(url, echo=echo)


async def async_connection_db(
    engine: AsyncEngine,
    expire_on_commit: bool = True,
) -> AsyncSession:
    """
    Create an async database session.

    Args:
        engine (AsyncEngine): The async database engine.
        expire_on_commit (bool): Whether to expire objects on commit. Defaults to True. When True, all instances will be fully expired after each commit(), so that all attribute/object access subsequent to a completed transaction will load from the most recent database state.

    Returns:
        AsyncSession: The async database session.
    """
    return async_sessionmaker(engine, expire_on_commit=expire_on_commit)


class RedisValueDTO(BaseModel):
    """Data Transfer Object (DTO) for Redis values."""

    key: str
    value: Optional[str]  # noqa: WPS110


def init_worker_redis() -> ConnectionPool:  # pragma: no cover
    """
    Create a connection pool for Redis.

    Returns:
        ConnectionPool: The Redis connection pool.
    """
    redis_pool: ConnectionPool = ConnectionPool.from_url(
        str(aiosettings.redis_url),
    )

    return redis_pool


def get_redis_conn_pool() -> ConnectionPool:  # pragma: no cover
    """
    Get the Redis connection pool.

    Returns:
        ConnectionPool: The Redis connection pool.
    """
    redis_pool: ConnectionPool = ConnectionPool.from_url(
        str(aiosettings.redis_url),
    )

    return redis_pool


async def shutdown_worker_redis(redis_pool: ConnectionPool) -> None:  # pragma: no cover
    """
    Close the Redis connection pool.

    Args:
        redis_pool (ConnectionPool): The Redis connection pool.
    """
    await redis_pool.disconnect()


async def get_redis_value(
    key: str,
    redis_pool: ConnectionPool,
) -> RedisValueDTO:
    """
    Get a value from Redis.

    Args:
        key (str): The Redis key.
        redis_pool (ConnectionPool): The Redis connection pool.

    Returns:
        RedisValueDTO: The Redis value DTO.
    """
    async with Redis(connection_pool=redis_pool) as redis:
        redis_value = await redis.get(key)
    return RedisValueDTO(
        key=key,
        value=redis_value,
    )


async def set_redis_value(redis_value: RedisValueDTO, redis_pool: ConnectionPool) -> None:
    """
    Set a value in Redis.

    Args:
        redis_value (RedisValueDTO): The Redis value DTO.
        redis_pool (ConnectionPool): The Redis connection pool.
    """
    if redis_value.value is not None:
        async with Redis(connection_pool=redis_pool) as redis:
            await redis.set(name=redis_value.key, value=redis_value.value)


if __name__ == "__main__":

    async def test_async_connection_db_smoke_test():
        db_session = await async_connection_db(
            engine=await create_async_engine_db(
                url=aiosettings.postgres_url,
                echo=True,
            ),
            expire_on_commit=True,
        )

        print(db_session)

    asyncio.run(test_async_connection_db_smoke_test())
