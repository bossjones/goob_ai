"""goob_ai sentry integration"""

from __future__ import annotations

from asyncio.exceptions import CancelledError
from typing import Any, Optional

from loguru import logger as LOGGER
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError, ResponseError
from sentry_sdk import HttpTransport
from sentry_sdk import init as sentry_sdk_init
from sentry_sdk.api import set_tag
from sentry_sdk.integrations.argv import ArgvIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.socket import SocketIntegration
from sentry_sdk.integrations.stdlib import StdlibIntegration
from sentry_sdk.integrations.threading import ThreadingIntegration
from websockets.exceptions import WebSocketException

from goob_ai.aio_settings import aiosettings, goob_user_agent


class SentryIgnoredException(Exception):
    """Base Class for all errors that are suppressed, and not sent to sentry."""


class SentryTransport(HttpTransport):
    """Custom sentry transport with custom user-agent"""

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        self._auth = self.parsed_dsn.to_auth(goob_user_agent())


def sentry_init(**sentry_init_kwargs):
    """Configure sentry SDK"""
    sentry_env = "bossjones"
    kwargs = {
        "environment": sentry_env,
        "send_default_pii": False,
        "_experiments": {
            "profiles_sample_rate": 0.1,
        },
    }
    kwargs.update(**sentry_init_kwargs)
    # pylint: disable=abstract-class-instantiated
    sentry_sdk_init(
        dsn=aiosettings.sentry_dsn,
        integrations=[
            ArgvIntegration(),
            StdlibIntegration(),
            RedisIntegration(),
            ThreadingIntegration(propagate_hub=True),
            SocketIntegration(),
        ],
        before_send=before_send,
        traces_sampler=traces_sampler,
        release=f"{goob_user_agent()}",
        transport=SentryTransport,
        **kwargs,
    )
    set_tag("goob_ai.build_hash", "")
    set_tag("goob_ai.env", "dev")
    set_tag("goob_ai.component", "backend")


def traces_sampler(sampling_context: dict) -> float:
    """Custom sampler to ignore certain routes"""
    path = sampling_context.get("asgi_scope", {}).get("path", "")
    _type = sampling_context.get("asgi_scope", {}).get("type", "")
    # Ignore all healthcheck routes
    if path.startswith("/-/health") or path.startswith("/-/metrics"):
        return 0
    if _type == "websocket":
        return 0
    return 0.1


def before_send(event: dict, hint: dict) -> Optional[dict]:
    """Check if error is database error, and ignore if so"""
    # pylint: disable=no-name-in-module
    ignored_classes = (
        # Inbuilt types
        KeyboardInterrupt,
        ConnectionResetError,
        OSError,
        PermissionError,
        # Redis errors
        RedisConnectionError,
        RedisError,
        ResponseError,
        # websocket errors
        WebSocketException,
        # custom baseclass
        SentryIgnoredException,
        # AsyncIO
        CancelledError,
    )
    exc_value = None
    if "exc_info" in hint:
        _, exc_value, _ = hint["exc_info"]
        if isinstance(exc_value, ignored_classes):
            LOGGER.debug("dropping exception", exc=exc_value)
            return None
    if "logger" in event:
        if event["logger"] in [
            # "kombu",
            "asyncio",
            "multiprocessing",
            # "django_redis",
            # "django.security.DisallowedHost",
            # "django_redis.cache",
            # "celery.backends.redis",
            # "celery.worker",
            "paramiko.transport",
        ]:
            return None
    LOGGER.debug("sending event to sentry", exc=exc_value, source_logger=event.get("logger", None))
    if aiosettings.debug:
        return None
    return event
