""" aio_settings """
from __future__ import annotations

import enum
import pathlib
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, Optional

from pydantic import BaseSettings, validator
from rich.console import Console
from rich.table import Table
from yarl import URL

# import tilda

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
    """Validator used to detect shell tilda notation and expand field automatically

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
    cerebro_root: str = "~/cerebro"
    monitor_host: str = "localhost"
    monitor_port: int = 50101

    # tweetpik_background_image = "510"  # a image that you want to use as background. you need to use this as a valid url like https://mysite.com/image.png and it should not be protected by cors
    audit_log_send_channel = ""

    # ***************************************************
    # NOTE: these are grouped together
    # ***************************************************
    token: str = ""
    prefix: str = "%"
    # default_config = {"token": "", "prefix": constants.prefix}
    # ***************************************************

    default_dropbox_folder: str = "/cerebro_downloads"
    # default_tweetpik_options = {}
    discord_admin_user_id: str = ""
    # TODO: ^ change this ->  to ^  and find all instances where it is called. discord_admin = os.environ.get("discord_admin_user_id")
    discord_general_channel: int = 908894727779258390

    discord_server_id: str = ""
    # TODO: ^ change this ->  to ^  and find all instances where it is called. discord_guild: str = "" os.environ.get("discord_server_id")

    # discord_server_id: Optional[str] = None
    discord_token: str = ""
    dropbox_cerebro_app_key: str = ""
    dropbox_cerebro_app_secret: str = ""
    dropbox_cerebro_token: str = ""
    tweetpik_authorization: str = ""
    # change the background color of the tweet screenshot
    tweetpik_background_color: str = "#ffffff"
    tweetpik_bucket_id: str = "323251495115948625"

    # any number higher than zero. this value is used in pixels(px) units
    tweetpik_canvas_width: str = "510"
    tweetpik_dimension_ig_feed: str = "1:1"
    tweetpik_dimension_ig_story: str = "9:16"
    tweetpik_display_likes: bool = False
    tweetpik_display_link_preview: bool = True
    tweetpik_display_media_images: bool = True
    tweetpik_display_replies: bool = False
    tweetpik_display_retweets: bool = False
    tweetpik_display_source: bool = True
    tweetpik_display_time: bool = True
    tweetpik_display_verified: bool = True

    # change the link colors used for the links, hashtags and mentions
    tweetpik_link_color: str = "#1b95e0"

    tweetpik_text_primary_color: str = (
        "#000000"  # change the text primary color used for the main text of the tweet and user's name
    )
    tweetpik_text_secondary_color: str = (
        "#5b7083"  # change the text secondary used for the secondary info of the tweet like the username
    )

    # any number higher than zero. this value is representing a percentage
    tweetpik_text_width: str = "100"

    tweetpik_timezone: str = "america/new_york"

    # change the verified icon color
    tweetpik_verified_icon: str = "#1b95e0"

    globals_root = cerebro_root

    # Where to look for the initialization file
    globals_initfile = "cerebro.init"
    globals_models_file = "models.yaml"
    globals_models_dir = "ml_models"
    globals_config_dir = "configs"
    globals_autoscan_dir = "weights"
    globals_converted_ckpts_dir = "converted_ckpts"

    # Try loading patchmatch
    globals_try_patchmatch = True

    # Use CPU even if GPU is available (main use case is for debugging MPS issues)
    globals_always_use_cpu = False

    # Whether the internet is reachable for dynamic downloads
    # The CLI will test connectivity at startup time.
    globals_internet_available = True

    # whether we are forcing full precision
    globals_full_precision = False

    # whether we should convert ckpt files into diffusers models on the fly
    globals_ckpt_convert = False

    # logging tokenization everywhere
    globals_log_tokenization = False

    gallery_dl_config_dir: str = "~/.config/gallery-dl"
    gallery_dl_homedir_config: str = "~/.gallery-dl.conf"
    netrc_config_filepath: str = "~/.netrc"

    yt_dl_config_dir: str = "~/Downloads"

    # TODO: Convert all of these to settings
    # DISCORD_SERVER_ID
    # DROPBOX_GOOB_AI_APP_KEY = os.environ.get("DROPBOX_GOOB_AI_APP_KEY")
    # DROPBOX_GOOB_AI_APP_SECRET = os.environ.get("DROPBOX_GOOB_AI_APP_SECRET")

    # DROPBOX_GOOB_AI_TOKEN = os.environ.get("DROPBOX_GOOB_AI_TOKEN")
    # DEFAULT_DROPBOX_FOLDER = "/cerebro_downloads"

    # DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
    # default_config = {"token": "", "prefix": constants.PREFIX}
    # AUDIT_LOG_SEND_CHANNEL = os.environ.get("AUDIT_LOG_SEND_CHANNEL")

    # TWEETPIK_AUTHORIZATION = os.environ.get("TWEETPIK_AUTHORIZATION")
    # TWEETPIK_BUCKET_ID = "323251495115948625"

    # TWEETPIK_DIMENSION_IG_FEED = "1:1"
    # TWEETPIK_DIMENSION_IG_STORY = "9:16"
    # TWEETPIK_TIMEZONE = "America/New_York"
    # TWEETPIK_DISPLAY_LIKES = False
    # TWEETPIK_DISPLAY_REPLIES = False
    # TWEETPIK_DISPLAY_RETWEETS = False
    # TWEETPIK_DISPLAY_VERIFIED = True
    # TWEETPIK_DISPLAY_SOURCE = True
    # TWEETPIK_DISPLAY_TIME = True
    # TWEETPIK_DISPLAY_MEDIA_IMAGES = True
    # TWEETPIK_DISPLAY_LINK_PREVIEW = True
    # TWEETPIK_TEXT_WIDTH = (
    #     "100"  # Any number higher than zero. This value is representing a percentage
    # )
    # TWEETPIK_CANVAS_WIDTH = (
    #     "510"  # Any number higher than zero. This value is used in pixels(px) units
    # )
    # # TWEETPIK_BACKGROUND_IMAGE = "510"  # A image that you want to use as background. You need to use this as a valid URL like https://mysite.com/image.png and it should not be protected by CORS

    # TWEETPIK_BACKGROUND_COLOR = (
    #     "#FFFFFF"  # Change the background color of the tweet screenshot
    # )
    # TWEETPIK_TEXT_PRIMARY_COLOR = "#000000"  # Change the text primary color used for the main text of the tweet and user's name
    # TWEETPIK_TEXT_SECONDARY_COLOR = "#5B7083"  # Change the text secondary used for the secondary info of the tweet like the username
    # TWEETPIK_LINK_COLOR = (
    #     "#1B95E0"  # Change the link colors used for the links, hashtags and mentions
    # )
    # TWEETPIK_VERIFIED_ICON = "#1B95E0"  # Change the verified icon color

    # DEFAULT_TWEETPIK_OPTIONS = {}

    # DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
    # DISCORD_ADMIN = os.environ.get("DISCORD_ADMIN_USER_ID")
    # DISCORD_GUILD = os.environ.get("DISCORD_SERVER_ID")
    # DISCORD_GENERAL_CHANNEL = 908894727779258390

    # ************************************************************************

    log_level: LogLevel = LogLevel.DEBUG

    # Variables for Redis
    redis_host: str = "localhost"
    redis_port: int = 7600
    redis_user: Optional[str] = None
    redis_pass: Optional[str] = None
    redis_base: Optional[int] = None

    # validators
    _normalize_cerebro_root = validator("cerebro_root", allow_reuse=True)(normalize_settings_path)
    _normalize_gallery_dl_config_dir = validator("gallery_dl_config_dir", allow_reuse=True)(normalize_settings_path)
    _normalize_gallery_dl_homedir_config = validator("gallery_dl_homedir_config", allow_reuse=True)(
        normalize_settings_path
    )
    _normalize_netrc_config_filepath = validator("netrc_config_filepath", allow_reuse=True)(normalize_settings_path)
    _normalize_yt_dl_config_dir = validator("yt_dl_config_dir", allow_reuse=True)(normalize_settings_path)
    _normalize_globals_root = validator("globals_root", allow_reuse=True)(normalize_settings_path)

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

    @property
    def global_config_file(self) -> Path:
        """_summary_

        Returns:
            Path: _description_
        """
        return Path(self.globals_root, self.globals_config_dir, self.globals_models_file)

    @property
    def global_config_dir(self) -> Path:
        """_summary_

        Returns:
            Path: _description_
        """
        return Path(self.globals_root, self.globals_config_dir)

    @property
    def global_models_dir(self) -> Path:
        """_summary_

        Returns:
            Path: _description_
        """
        return Path(self.globals_root, self.globals_models_dir)

    @property
    def global_autoscan_dir(self) -> Path:
        """_summary_

        Returns:
            Path: _description_
        """
        return Path(self.globals_root, self.globals_autoscan_dir)

    @property
    def global_converted_ckpts_dir(self) -> Path:
        """_summary_

        Returns:
            Path: _description_
        """
        return Path(self.global_models_dir, self.globals_converted_ckpts_dir)

    @property
    def esgran_dir(self) -> Path:
        """_summary_

        Returns:
            Path: _description_
        """
        return Path(self.global_models_dir, "esgran")

    @property
    def screencropnet_dir(self) -> Path:
        """_summary_

        Returns:
            Path: _description_
        """
        return Path(self.global_models_dir, "screencropnet")

    @property
    def screencropnet_model_fpath(self) -> Path:
        """_summary_

        Returns:
            Path: _description_
        """
        return Path(self.screencropnet_dir, "ScreenCropNetV1_378_epochs.pth")

    @property
    def gallery_dl_config_dot_json(self) -> Path:
        """_summary_

        Returns:
            Path: _description_
        """
        return Path(self.gallery_dl_config_dir, "config.json")

    @property
    def yt_dlp_config(self) -> Path:
        """_summary_

        Returns:
            Path: _description_
        """
        return Path(self.yt_dl_config_dir, "yt-cookies.txt")

    @property
    def netrc_config(self) -> Path:
        """_summary_

        Returns:
            Path: _description_
        """
        return Path(self.netrc_config_filepath)

    @property
    def gallery_dl_cookie_files(self) -> Dict[str, Path]:
        """_summary_

        Returns:
            Path: _description_
        """
        return {
            "instagram": Path(self.gallery_dl_config_dir, "cookies-instagram.txt"),
            "pinterest": Path(self.gallery_dl_config_dir, "cookies-pinterest.txt"),
            "reddit": Path(self.gallery_dl_config_dir, "cookies-reddit.txt"),
            "redgifs": Path(self.gallery_dl_config_dir, "cookies-redgifs.txt"),
            "twitter": Path(self.gallery_dl_config_dir, "cookies-twitter.txt"),
            "default_cookies": Path(self.gallery_dl_config_dir, "cookies.txt"),
            "hlm_ig": Path(self.gallery_dl_config_dir, "hlm-cookies-instagram.txt"),
            "reactionmemestv_ig": Path(self.gallery_dl_config_dir, "reactionmemestv-cookies-instagram.txt"),
            "universityofprofessorex_ig": Path(
                self.gallery_dl_config_dir, "universityofprofessorex-cookies-instagram.txt"
            ),
            "wavy_ig": Path(self.gallery_dl_config_dir, "wavy-cookies-instagram.txt"),
        }

    class Config:  # sourcery skip: docstrings-for-classes
        env_file = ".env"
        env_prefix = "CEREBROBOT_CONFIG_"
        env_file_encoding = "utf-8"


aiosettings = AioSettings()  # sourcery skip: docstrings-for-classes, avoid-global-variables

# Need to create a command that can audit and make sure all of these are available.
# pi@boss-deeplearning ~/dev/just master*
# ❯ tree ~/.config/gallery-dl/
# /home/pi/.config/gallery-dl/
# ├── config.json -> /home/pi/.gallery-dl.conf
# ├── cookies-instagram.txt
# ├── cookies-pinterest.txt
# ├── cookies-reddit.txt
# ├── cookies-redgifs.txt
# ├── cookies-twitter.txt
# ├── cookies.txt
# ├── cookies.txt.bak
# ├── hlm-cookies-instagram.txt
# ├── reactionmemestv-cookies-instagram.txt
# ├── universityofprofessorex-cookies-instagram.txt
# └── wavy-cookies-instagram.txt

# /home/pi/.netrc
# ls -lta ~/.gallery-dl.conf

# pi@boss-deeplearning ~/dev/just master*
# ❯ ls -lta ~/Downloads/yt-cookies.txt
# -rw-r--r-- 1 pi pi 9626 Mar  4 16:42 /home/pi/Downloads/yt-cookies.txt
# -rw-r--r-- 1 pi pi 9626 Mar  4 16:42 /home/pi/Downloads/yt-cookies.txt
# -rw-r--r-- 1 pi pi 9626 Mar  4 16:42 /home/pi/Downloads/yt-cookies.txt
