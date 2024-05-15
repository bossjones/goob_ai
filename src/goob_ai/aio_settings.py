"""aio_settings"""
# pylint: disable=no-name-in-module

from __future__ import annotations

import enum
import pathlib

from pathlib import Path
from tempfile import gettempdir
from typing import Any, Callable, Dict, List, Optional, Set, Union, cast

from pydantic import AliasChoices, AmqpDsn, BaseModel, Field, ImportString, PostgresDsn, RedisDsn

# from pydantic.functional_validators import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated, TypedDict
from yarl import URL


TEMP_DIR = Path(gettempdir())


# NOTE: DIRTY HACK TO GET AROUND CIRCULAR IMPORTS
# NOTE: There is a bug in pydantic that prevents us from using the `tilda` package and dealing with circular imports
def tilda(obj):
    """wrapper for linux ~/ shell notation

    Args:
        obj (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(obj, list):
        return [str(pathlib.Path(o).expanduser()) if isinstance(o, str) else o for o in obj]
    elif isinstance(obj, str):
        return str(pathlib.Path(obj).expanduser())
    else:
        return obj


def normalize_settings_path(file_path: str) -> str:
    """field_validator used to detect shell tilda notation and expand field automatically

    Args:
        file_path (str): _description_

    Returns:
        pathlib.PosixPath | str: _description_
    """
    # prevent circular import
    # from goob_ai.utils import file_functions

    return tilda(file_path) if file_path.startswith("~") else file_path


def get_rich_console() -> Console:
    """_summary_

    Returns:
        Console: _description_
    """
    return Console()


class LogLevel(str, enum.Enum):  # noqa: WPS600
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class AioSettings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    # By default, the environment variable name is the same as the field name.

    # You can change the prefix for all environment variables by setting the env_prefix config setting, or via the _env_prefix keyword argument on instantiation:

    model_config = SettingsConfigDict(
        env_prefix="GOOB_AI_CONFIG_", env_file=(".env", ".envrc"), env_file_encoding="utf-8"
    )

    monitor_host: str = "localhost"
    monitor_port: int = 50102

    # tweetpik_background_image = "510"  # a image that you want to use as background. you need to use this as a valid url like https://mysite.com/image.png and it should not be protected by cors
    audit_log_send_channel: str = ""

    # ***************************************************
    # NOTE: these are grouped together
    # ***************************************************
    # token: str = ""
    prefix: str = "?"

    discord_admin_user_id: int | None = None

    discord_general_channel: int = 908894727779258390

    discord_server_id: int = 0
    discord_client_id: int | str = 0

    discord_token: str = ""

    # openai_token: str = ""
    openai_api_key: str = ""

    discord_admin_user_invited: bool = False

    debug: bool = True

    # pylint: disable=redundant-keyword-arg
    better_exceptions: bool = Field(env="BETTER_EXCEPTIONS", description="Enable better exceptions", default=1)
    pythonasynciodebug: bool = Field(
        env="PYTHONASYNCIODEBUG", description="enable or disable asyncio debugging", default=1
    )
    langchain_debug_logs: bool = Field(
        env="LANGCHAIN_DEBUG_LOGS", description="enable or disable langchain debug logs", default=0
    )

    enable_ai: bool = False
    http_client_debug_enabled: bool = False

    # Try loading patchmatch
    globals_try_patchmatch: bool = True

    # Use CPU even if GPU is available (main use case is for debugging MPS issues)
    globals_always_use_cpu: bool = False

    # Whether the internet is reachable for dynamic downloads
    # The CLI will test connectivity at startup time.
    globals_internet_available: bool = True

    # whether we are forcing full precision
    globals_full_precision: bool = False

    # whether we should convert ckpt files into diffusers models on the fly
    globals_ckpt_convert: bool = False

    # logging tokenization everywhere
    globals_log_tokenization: bool = False

    # Variables for Redis
    redis_host: str = "localhost"
    redis_port: int = 7600
    redis_user: Optional[str] = None
    redis_pass: Optional[str] = None
    redis_base: Optional[int] = None
    enable_redis: bool = False

    # azure_openai_api_key: str
    # openai_api_type: str
    # openai_api_version: str
    # azure_deployment: str
    # azure_openai_endpoint: str
    llm_temperature: float = 0.1
    # vision_model: str = "gpt-4-turbo"
    vision_model: str = "gpt-4-vision-preview"
    chat_model: str = "gpt-3.5-turbo-0125"
    # chat_model: str = "gpt-3.5-turbo-16k" # note another option
    chat_history_buffer: int = 10

    retry_stop_after_attempt: int = 3
    retry_wait_exponential_multiplier: Union[int, float] = 2
    retry_wait_exponential_max: Union[int, float] = 5
    retry_wait_exponential_min: Union[int, float] = 1
    retry_wait_fixed: Union[int, float] = 15

    pinecone_api_key: str = Field(env="PINECONE_API_KEY", description="pinecone api key", default="")
    pinecone_env: str = Field(env="PINECONE_ENV", description="pinecone env", default="")
    pinecone_index: str = Field(env="PINECONE_INDEX", description="pinecone index", default="")

    @property
    def redis_url(self) -> URL:
        """
        Assemble REDIS URL from settings.

        :return: redis URL.
        """
        path = f"/{self.redis_base}" if self.redis_base is not None else ""
        return URL.build(
            scheme="redis",
            host=self.redis_host,
            port=self.redis_port,
            user=self.redis_user,
            password=self.redis_pass,
            path=path,
        )


aiosettings = AioSettings()  # sourcery skip: docstrings-for-classes, avoid-global-variables
