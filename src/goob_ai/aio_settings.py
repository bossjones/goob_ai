"""aio_settings"""

# pylint: disable=no-name-in-module
# type: ignore[call-arg]
# type: ignore[attr-defined]
# type: ignore[pydantic-field]
from __future__ import annotations

from typing import Any, Callable, List, cast
from typing import Any, Callable, Set

from pydantic import (
    AliasChoices,
    AmqpDsn,
    BaseModel,
    Field,
    ImportString,
    PostgresDsn,
    RedisDsn,
)

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated, TypedDict

import enum
import pathlib
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic.functional_validators import field_validator
from rich.console import Console
from rich.table import Table
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


def add_rich_table_row(table: Table, config_name: str, config_value: Any, bot_settings: AioSettings) -> None:
    """Format and add row to Rich table.

    Args:
        table (Table): _description_
        config_name (str): _description_
        config_value (Any): _description_
        bot_settings (AioSettings): _description_
    """
    formatted_config_name = f"{config_name}"
    formatted_config_value: str = f"{config_value}"
    env_name = f"{bot_settings.Config.env_prefix}{config_name}".upper()
    default_value = f"{bot_settings.__fields__[config_name].default}"
    config_type: str = f"{bot_settings.__fields__[config_name].type_}"
    table.add_row(formatted_config_name, env_name, formatted_config_value, config_type, default_value)


def config_to_table(console: Console, bot_settings: AioSettings) -> None:
    """_summary_

    Args:
        console (Console): _description_
        bot_settings (AioSettings): _description_
    """
    # config name, env var name, Value, default
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Config Name", style="dim")
    table.add_column("Env Var Name")
    table.add_column("Value")
    table.add_column("Type")
    table.add_column("Default Value")

    # Iterating over values
    for config_name, config_value in bot_settings.dict().items():
        add_rich_table_row(table, config_name, config_value, bot_settings)

    # # handle properties differently.
    table.add_row("redis_url", "n/a", f"{bot_settings.redis_url}", f"{type(bot_settings.redis_url)}", "n/a")
    table.add_row(
        "aiomonitor_config_data",
        "n/a",
        f"{bot_settings.aiomonitor_config_data}",
        f"{type(bot_settings.aiomonitor_config_data)}",
        "n/a",
    )
    table.add_row(
        "global_config_file",
        "n/a",
        f"{bot_settings.global_config_file}",
        f"{type(bot_settings.global_config_file)}",
        "n/a",
    )
    table.add_row(
        "global_config_dir",
        "n/a",
        f"{bot_settings.global_config_dir}",
        f"{type(bot_settings.global_config_dir)}",
        "n/a",
    )
    table.add_row(
        "global_models_dir",
        "n/a",
        f"{bot_settings.global_models_dir}",
        f"{type(bot_settings.global_models_dir)}",
        "n/a",
    )
    table.add_row(
        "global_autoscan_dir",
        "n/a",
        f"{bot_settings.global_autoscan_dir}",
        f"{type(bot_settings.global_autoscan_dir)}",
        "n/a",
    )
    table.add_row(
        "global_converted_ckpts_dir",
        "n/a",
        f"{bot_settings.global_converted_ckpts_dir}",
        f"{type(bot_settings.global_converted_ckpts_dir)}",
        "n/a",
    )

    console.print(table)


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

    # main directory to place models in etc eg. ~/cerebro
    # cerebro_root: str = "~/cerebro"
    monitor_host: str = "localhost"
    monitor_port: int = 50102

    # tweetpik_background_image = "510"  # a image that you want to use as background. you need to use this as a valid url like https://mysite.com/image.png and it should not be protected by cors
    audit_log_send_channel: str = ""

    # ***************************************************
    # NOTE: these are grouped together
    # ***************************************************
    token: str = ""
    prefix: str = "/"
    # default_config = {"token": "", "prefix": constants.prefix}
    # ***************************************************

    # default_dropbox_folder: str = "/cerebro_downloads"
    # default_tweetpik_options = {}
    discord_admin_user_id: str = ""
    # TODO: ^ change this ->  to ^  and find all instances where it is called. discord_admin = os.environ.get("discord_admin_user_id")
    discord_general_channel: int = 908894727779258390

    discord_server_id: str = ""
    # TODO: ^ change this ->  to ^  and find all instances where it is called. discord_guild: str = "" os.environ.get("discord_server_id")

    # discord_server_id: Optional[str] = None
    discord_token: str = ""

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

    # ************************************************************************

    # log_level: LogLevel = LogLevel.DEBUG.value
    # log_level

    # Variables for Redis
    redis_host: str = "localhost"
    redis_port: int = 7600
    redis_user: Optional[str] = None
    redis_pass: Optional[str] = None
    redis_base: Optional[int] = None

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

    @property
    def aiomonitor_config_data(self) -> Dict:
        """_summary_

        Returns:
            Path: _description_
        """
        return {"port": self.monitor_port, "host": self.monitor_host}

    class Config:  # sourcery skip: docstrings-for-classes
        env_file = ".env"
        env_prefix = "GOOB_AI_"
        env_file_encoding = "utf-8"


aiosettings = AioSettings()  # sourcery skip: docstrings-for-classes, avoid-global-variables
