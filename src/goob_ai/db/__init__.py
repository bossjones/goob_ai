"""goob_ai.db"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from redis.asyncio import ConnectionPool, Redis

from goob_ai.aio_settings import aiosettings


class RedisValueDTO(BaseModel):
    """Data Transfer Object(DTO) for redis values."""

    key: str
    value: Optional[str]  # noqa: WPS110


def init_worker_redis() -> ConnectionPool:  # pragma: no cover
    """
    Creates connection pool for redis.
    """
    redis_pool: ConnectionPool = ConnectionPool.from_url(
        str(aiosettings.redis_url),
    )

    return redis_pool


async def shutdown_worker_redis(redis_pool: ConnectionPool) -> None:  # pragma: no cover
    """
    Closes redis connection pool.

    :param redis_pool: redis ConnectionPool.
    """
    await redis_pool.disconnect()


# get from redis
# async with Redis(connection_pool=redis_pool) as redis:
#     # exists = await redis.hexists(inference_id, "pred_prob")
#     exists = await redis.exists(inference_id)
#     if not exists:
#         pending_classification_dto = {"inference_id": inference_id}
#         # return PendingClassificationDTO(inference_id=inference_id)
#         return JSONResponse(
#             status_code=status.HTTP_202_ACCEPTED, content=pending_classification_dto,
#         )
#         # return status.HTTP_202_ACCEPTED

#     redis_value = await redis.hgetall(inference_id)
# return RedisPredictionValueDTO(data=redis_value)


async def get_redis_value(
    key: str,
    redis_pool: ConnectionPool,
) -> None:
    """
    Get value from redis.

    :param key: redis key, to get data from.
    :param redis_pool: redis connection pool.
    :returns: information from redis.
    """
    async with Redis(connection_pool=redis_pool) as redis:
        redis_value = await redis.get(key)
    return RedisValueDTO(
        key=key,
        value=redis_value,
    )


async def set_redis_value(redis_value: RedisValueDTO, redis_pool: ConnectionPool) -> None:
    """
    Set value in redis.

    :param redis_value: new value data.
    :param redis_pool: redis connection pool.
    """
    if redis_value.value is not None:
        async with Redis(connection_pool=redis_pool) as redis:
            await redis.set(name=redis_value.key, value=redis_value.value)
