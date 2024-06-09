"""goob_ai.goob_bot"""

# pylint: disable=possibly-used-before-assignment
# pyright: reportImportCycles=false
from __future__ import annotations

import asyncio
import base64
import datetime
import io
import json
import logging
import os
import pathlib
import re
import sys
import tempfile
import time
import traceback
import typing
import uuid

from collections import Counter, defaultdict
from io import BytesIO
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    List,
    NoReturn,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import aiohttp
import discord
import rich

from codetiming import Timer
from discord.ext import commands
from langchain.text_splitter import RecursiveCharacterTextSplitter
from logging_tree import printout
from loguru import logger as LOGGER
from PIL import Image
from redis.asyncio import ConnectionPool as RedisConnectionPool
from starlette.responses import JSONResponse, StreamingResponse

import goob_ai

from goob_ai import db, helpers, shell, utils
from goob_ai.agent import AiAgent
from goob_ai.aio_settings import aiosettings
from goob_ai.bot_logger import REQUEST_ID_CONTEXTVAR, generate_tree, get_lm_from_tree, get_logger
from goob_ai.common.dataclasses import SurfaceInfo, SurfaceType
from goob_ai.constants import CHANNEL_ID, INPUT_CLASSIFICATION_NOT_A_QUESTION, INPUT_CLASSIFICATION_NOT_FOR_ME
from goob_ai.factories import guild_factory
from goob_ai.gen_ai.utilities.agent_criteria_evaluator import Evaluator
from goob_ai.user_input_enrichment import UserInputEnrichment
from goob_ai.utils import async_, file_functions
from goob_ai.utils.context import Context
from goob_ai.utils.misc import CURRENTFUNCNAME


LOGGER.add(sys.stderr, level="DEBUG")

DESCRIPTION = """An example bot to showcase the discord.ext.commands extension
module.

There are a number of utility commands being showcased here."""

HERE = os.path.dirname(__file__)

INVITE_LINK = "https://discordapp.com/api/oauth2/authorize?client_id={}&scope=bot&permissions=0"


HOME_PATH = os.environ.get("HOME")

COMMAND_RUNNER = {"dl_thumb": shell.run_coroutine_subprocess}


def unlink_orig_file(a_filepath: str) -> str:
    """
    Delete the specified file and return its path.

    This function deletes the file at the given file path and returns the path of the deleted file.
    It uses the `os.unlink` method to remove the file and logs the deletion using `rich.print`.

    Args:
        a_filepath (str): The path to the file to be deleted.

    Returns:
        str: The path of the deleted file.
    """
    rich.print(f"deleting ... {a_filepath}")
    os.unlink(f"{a_filepath}")
    return a_filepath


# https://github.com/discord-math/bot/blob/babb41b71a68b4b099684b3e1ed583f84083f971/plugins/log.py#L63
def path_for(attm: discord.Attachment, basedir: str = "./") -> pathlib.Path:
    """Generate a pathlib.Path object for an attachment with a specified base directory.

    This function constructs a pathlib.Path object for a given attachment 'attm' using the specified base directory 'basedir'.
    It logs the generated path for debugging purposes and returns the pathlib.Path object.

    Args:
        attm (discord.Attachment): The attachment for which the path is generated.
        basedir (str): The base directory path where the attachment file will be located. Default is the current directory.

    Returns:
        pathlib.Path: A pathlib.Path object representing the path for the attachment file.

    Example:
        >>> attm = discord.Attachment(filename="example.png")
        >>> path = path_for(attm, basedir="/attachments")
        >>> print(path)
        /attachments/example.png
    """
    p = pathlib.Path(basedir, str(attm.filename))  # pyright: ignore[reportAttributeAccessIssue]
    LOGGER.debug(f"path_for: p -> {p}")
    return p


# https://github.com/discord-math/bot/blob/babb41b71a68b4b099684b3e1ed583f84083f971/plugins/log.py#L63
async def save_attachment(attm: discord.Attachment, basedir: str = "./") -> None:
    """Save a Discord attachment to a specified directory.

    This asynchronous function saves a Discord attachment to the specified base directory.
    It constructs the path for the attachment, creates the necessary directories, and saves the attachment
    to the generated path. If an HTTPException occurs during saving, it retries the save operation.

    Args:
        attm (discord.Attachment): The attachment to be saved.
        basedir (str): The base directory path where the attachment file will be located. Default is the current directory.

    Returns:
        None
    """
    path = path_for(attm, basedir=basedir)
    LOGGER.debug(f"save_attachment: path -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ret_code = await attm.save(path, use_cached=True)
        await asyncio.sleep(5)
    except discord.HTTPException:
        await attm.save(path)


def attachment_to_dict(attm: discord.Attachment) -> Dict[str, Any]:
    """Convert a discord.Attachment object to a dictionary.

    This function takes a discord.Attachment object and converts it into a dictionary
    containing relevant information about the attachment.

    The dictionary includes:
    - filename: The name of the file.
    - id: The unique identifier of the attachment.
    - proxy_url: The proxy URL of the attachment.
    - size: The size of the attachment in bytes.
    - url: The URL of the attachment.
    - spoiler: A boolean indicating if the attachment is a spoiler.
    - height: The height of the attachment (if applicable).
    - width: The width of the attachment (if applicable).
    - content_type: The MIME type of the attachment (if applicable).
    - attachment_obj: The original attachment object.

    Args:
        attm (discord.Attachment): The attachment object to be converted.

    Returns:
        Dict[str, Any]: A dictionary containing information about the attachment.
    """
    result = {
        "filename": attm.filename,  # pyright: ignore[reportAttributeAccessIssue]
        "id": attm.id,
        "proxy_url": attm.proxy_url,  # pyright: ignore[reportAttributeAccessIssue]
        "size": attm.size,  # pyright: ignore[reportAttributeAccessIssue]
        "url": attm.url,  # pyright: ignore[reportAttributeAccessIssue]
        "spoiler": attm.is_spoiler(),
    }
    if attm.height:  # pyright: ignore[reportAttributeAccessIssue]
        result["height"] = attm.height  # pyright: ignore[reportAttributeAccessIssue]
    if attm.width:  # pyright: ignore[reportAttributeAccessIssue]
        result["width"] = attm.width  # pyright: ignore[reportAttributeAccessIssue]
    if attm.content_type:  # pyright: ignore[reportAttributeAccessIssue]
        result["content_type"] = attm.content_type  # pyright: ignore[reportAttributeAccessIssue]

    result["attachment_obj"] = attm

    return result


def file_to_local_data_dict(fname: str, dir_root: str) -> Dict[str, Any]:
    """Convert a file to a dictionary with metadata.

    This function takes a file path and a root directory, and converts the file
    into a dictionary containing metadata such as filename, size, extension, and
    a pathlib.Path object representing the file.

    Args:
        fname (str): The name of the file to be converted.
        dir_root (str): The root directory where the file is located.

    Returns:
        Dict[str, Any]: A dictionary containing metadata about the file.
    """
    file_api = pathlib.Path(fname)
    return {
        "filename": f"{dir_root}/{file_api.stem}{file_api.suffix}",
        "size": file_api.stat().st_size,
        "ext": f"{file_api.suffix}",
        "api": file_api,
    }


async def handle_save_attachment_locally(attm_data_dict: Dict[str, Any], dir_root: str) -> str:
    """Save a Discord attachment locally.

    This asynchronous function saves a Discord attachment to a specified directory.
    It constructs the file path for the attachment, saves the attachment to the generated path,
    and returns the path of the saved file.

    Args:
        attm_data_dict (Dict[str, Any]): A dictionary containing information about the attachment.
        dir_root (str): The root directory where the attachment file will be saved.

    Returns:
        str: The path of the saved attachment file.
    """
    fname = f"{dir_root}/orig_{attm_data_dict['id']}_{attm_data_dict['filename']}"
    rich.print(f"Saving to ... {fname}")
    await attm_data_dict["attachment_obj"].save(fname, use_cached=True)
    await asyncio.sleep(1)
    return fname


# SOURCE: https://github.com/CrosswaveOmega/NikkiBot/blob/75c7ecd307f50390cfc798d39098fdb78535650c/cogs/AiCog.py#L237
async def download_image(url: str) -> BytesIO:
    """
    Download an image from a given URL asynchronously.

    This asynchronous function uses aiohttp to make a GET request to the provided URL
    and downloads the image data. If the response status is 200 (OK), it reads the
    response data and returns it as a BytesIO object.

    Args:
        url (str): The URL of the image to download.

    Returns:
        BytesIO: A BytesIO object containing the downloaded image data.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.read()
                return io.BytesIO(data)


# SOURCE: https://github.com/CrosswaveOmega/NikkiBot/blob/7092ae6da21c86c7686549edd5c45335255b73ec/cogs/GlobalCog.py#L23
async def file_to_data_uri(file: discord.File) -> str:
    """Convert a discord.File object to a data URI.

    This asynchronous function reads the bytes from a discord.File object,
    base64 encodes the bytes, and constructs a data URI.

    Args:
        file (discord.File): The discord.File object to be converted.

    Returns:
        str: A data URI representing the file content.
    """
    with BytesIO(file.fp.read()) as f:
        file_bytes = f.read()
    base64_encoded = base64.b64encode(file_bytes).decode("ascii")
    data_uri = f'data:{"image"};base64,{base64_encoded}'
    return data_uri


# SOURCE: https://github.com/CrosswaveOmega/NikkiBot/blob/7092ae6da21c86c7686549edd5c45335255b73ec/cogs/GlobalCog.py#L23
async def data_uri_to_file(data_uri: str, filename: str) -> discord.File:
    """Convert a data URI to a discord.File object.

    This asynchronous function takes a data URI and a filename, decodes the base64 data,
    and creates a discord.File object with the decoded data.

    Args:
        data_uri (str): The data URI to be converted.
        filename (str): The name of the file to be created.

    Returns:
        discord.File: A discord.File object containing the decoded data.
    """
    # Split the data URI into its components
    metadata, base64_data = data_uri.split(",")
    # Get the content type from the metadata
    content_type = metadata.split(";")[0].split(":")[1]
    # Decode the base64 data
    file_bytes = base64.b64decode(base64_data)
    # Create a discord.File object
    file = discord.File(BytesIO(file_bytes), filename=filename, spoiler=False)
    return file


@async_.to_async
def get_logger_tree_printout() -> None:
    """Print the logger tree structure.

    This function prints the logger tree structure using the `printout` function
    from the `logging_tree` module. It is decorated with `@async_.to_async` to
    run asynchronously.
    """
    printout()


# SOURCE: https://realpython.com/how-to-make-a-discord-bot-python/#responding-to-messages
def dump_logger_tree() -> None:
    """Dump the logger tree structure.

    This function generates the logger tree structure using the `generate_tree` function
    and logs the tree structure using the `LOGGER.debug` method.
    """
    rootm = generate_tree()
    LOGGER.debug(rootm)


def dump_logger(logger_name: str) -> Any:
    """Dump the logger tree structure for a specific logger.

    This function generates the logger tree structure using the `generate_tree` function
    and retrieves the logger metadata for the specified logger name using the `get_lm_from_tree` function.
    It logs the retrieval process using the `LOGGER.debug` method.

    Args:
        logger_name (str): The name of the logger to retrieve the tree structure for.

    Returns:
        Any: The logger metadata for the specified logger name.
    """
    LOGGER.debug(f"getting logger {logger_name}")
    rootm = generate_tree()
    return get_lm_from_tree(rootm, logger_name)


def filter_empty_string(a_list: List[str]) -> List[str]:
    """Filter out empty strings from a list of strings.

    This function takes a list of strings and returns a new list with all empty strings removed.

    Args:
        a_list (List[str]): The list of strings to be filtered.

    Returns:
        List[str]: A new list containing only non-empty strings from the input list.
    """
    filter_object = filter(lambda x: x != "", a_list)
    return list(filter_object)


# SOURCE: https://docs.python.org/3/library/asyncio-queue.html
async def worker(name: str, queue: asyncio.Queue) -> NoReturn:
    """Process tasks from the queue.

    This asynchronous function continuously processes tasks from the provided queue.
    Each task is executed using the COMMAND_RUNNER dictionary, and the function
    notifies the queue when a task is completed.

    Args:
        name (str): The name of the worker.
        queue (asyncio.Queue): The queue from which tasks are retrieved.

    Returns:
        NoReturn: This function runs indefinitely and does not return.
    """
    LOGGER.info(f"starting working ... {name}")

    while True:
        # Get a "work item" out of the queue.
        co_cmd_task = await queue.get()
        print(f"co_cmd_task = {co_cmd_task}")

        # Sleep for the "co_cmd_task" seconds.

        await COMMAND_RUNNER[co_cmd_task.name](cmd=co_cmd_task.cmd, uri=co_cmd_task.uri)

        # Notify the queue that the "work item" has been processed.
        queue.task_done()

        print(f"{name} ran {co_cmd_task.name} with arguments {co_cmd_task}")


# SOURCE: https://realpython.com/python-async-features/#building-a-synchronous-web-server
async def co_task(name: str, queue: asyncio.Queue) -> AsyncIterator[None]:
    """
    Process tasks from the queue with timing.

    This asynchronous function processes tasks from the provided queue. It uses a timer to measure the elapsed time for each task. The function yields control back to the event loop after processing each task.

    Args:
        name (str): The name of the task.
        queue (asyncio.Queue): The queue from which tasks are retrieved.

    Yields:
        None: This function yields control back to the event loop after processing each task.
    """
    LOGGER.info(f"starting working ... {name}")

    timer = Timer(text=f"Task {name} elapsed time: {{:.1f}}")
    while not queue.empty():
        co_cmd_task = await queue.get()
        print(f"Task {name} running")
        timer.start()
        await COMMAND_RUNNER[co_cmd_task.name](cmd=co_cmd_task.cmd, uri=co_cmd_task.uri)
        timer.stop()
        yield


# SOURCE: https://github.com/makupi/cookiecutter-discord.py-postgres/blob/master/%7B%7Bcookiecutter.bot_slug%7D%7D/bot/__init__.py
async def get_prefix(_bot: AsyncGoobBot, message: discord.Message) -> Any:
    """
    Retrieve the command prefix for the bot based on the message context.

    This function determines the appropriate command prefix for the bot to use
    based on whether the message is from a direct message (DM) channel or a guild
    (server) channel. If the message is from a DM channel, it uses the default prefix
    from the settings. If the message is from a guild channel, it retrieves the prefix
    specific to that guild.

    Args:
        _bot (AsyncGoobBot): The instance of the bot.
        message (discord.Message): The message object from Discord.

    Returns:
        Any: The command prefix to be used for the bot.
    """
    LOGGER.info(f"inside get_prefix(_bot, message) - > get_prefix({_bot}, {message})")
    LOGGER.info(f"inside get_prefix(_bot, message) - > get_prefix({_bot}, {message})")
    LOGGER.info(f"inside get_prefix(_bot, message) - > get_prefix({type(_bot)}, {type(message)})")
    prefix = (
        [aiosettings.prefix]
        if isinstance(message.channel, discord.DMChannel)  # pyright: ignore[reportAttributeAccessIssue]
        else [utils.get_guild_prefix(_bot, message.guild.id)]  # type: ignore
    )
    LOGGER.info(f"prefix -> {prefix}")
    return commands.when_mentioned_or(*prefix)(_bot, message)


# SOURCE: https://github.com/makupi/cookiecutter-discord.py-postgres/blob/master/%7B%7Bcookiecutter.bot_slug%7D%7D/bot/__init__.py#L28
async def preload_guild_data() -> Dict[int, Dict[str, str]]:
    """
    Preload guild data.

    This function initializes and returns a dictionary containing guild data.
    Each guild is represented by its ID and contains a dictionary with the guild's prefix.

    Returns:
        Dict[int, Dict[str, str]]: A dictionary where the keys are guild IDs and the values are dictionaries
        containing guild-specific data, such as the prefix.
    """
    LOGGER.info("preload_guild_data ... ")
    guilds = [guild_factory.Guild()]
    return {guild.id: {"prefix": guild.prefix} for guild in guilds}


def extensions() -> Iterable[str]:
    """Yield extension module paths.

    This function searches for Python files in the 'cogs' directory relative to the current file's directory.
    It constructs the module path for each file and yields it.

    Yields:
        str: The module path for each Python file in the 'cogs' directory.
    """
    module_dir = pathlib.Path(HERE)
    files = pathlib.Path(module_dir.stem, "cogs").rglob("*.py")
    for file in files:
        LOGGER.debug(f"exension = {file.as_posix()[:-3].replace('/', '.')}")
        yield file.as_posix()[:-3].replace("/", ".")


def _prefix_callable(bot: AsyncGoobBot, msg: discord.Message) -> List[str]:
    """Generate a list of command prefixes for the bot.

    This function generates a list of command prefixes for the bot based on the message context.
    If the message is from a direct message (DM) channel, it includes the bot's user ID mentions
    and default prefixes. If the message is from a guild (server) channel, it includes the bot's
    user ID mentions and the guild-specific prefixes.

    Args:
        bot (AsyncGoobBot): The instance of the bot.
        msg (discord.Message): The message object from Discord.

    Returns:
        List[str]: A list of command prefixes to be used for the bot.
    """
    user_id = bot.user.id
    base = [f"<@!{user_id}> ", f"<@{user_id}> "]
    if msg.guild is None:  # pyright: ignore[reportAttributeAccessIssue]
        base.extend(("!", "?"))
    else:
        base.extend(bot.prefixes.get(msg.guild.id, ["?", "!"]))  # pyright: ignore[reportAttributeAccessIssue]
    return base


async def details_from_file(path_to_media_from_cli: str, cwd: typing.Union[str, None] = None) -> Tuple[str, str, str]:
    """Generate input and output file paths and retrieve the timestamp of the input file.

    This function takes a file path and an optional current working directory (cwd),
    and returns the input file path, output file path, and the timestamp of the input file.
    The timestamp is retrieved using platform-specific commands.

    Args:
        path_to_media_from_cli (str): The path to the media file provided via the command line.
        cwd (typing.Union[str, None], optional): The current working directory. Defaults to None.

    Returns:
        Tuple[str, str, str]: A tuple containing the input file path, output file path, and the timestamp of the input file.
    """
    p = pathlib.Path(path_to_media_from_cli)
    full_path_input_file = f"{p.stem}{p.suffix}"
    full_path_output_file = f"{p.stem}_smaller.mp4"
    rich.print(full_path_input_file)
    rich.print(full_path_output_file)
    if sys.platform == "darwin":
        get_timestamp = await shell._aio_run_process_and_communicate(
            ["gstat", "-c", "%y", f"{p.stem}{p.suffix}"], cwd=cwd
        )
    elif sys.platform == "linux":
        get_timestamp = await shell._aio_run_process_and_communicate(
            ["stat", "-c", "%y", f"{p.stem}{p.suffix}"], cwd=cwd
        )

    return full_path_input_file, full_path_output_file, get_timestamp


class ProxyObject(discord.Object):
    def __init__(self, guild: Optional[discord.abc.Snowflake]):
        super().__init__(id=0)
        self.guild: Optional[discord.abc.Snowflake] = guild


# class AsyncGoobBot(commands.AutoShardedBot):
class AsyncGoobBot(commands.Bot):
    user: discord.ClientUser
    # pool: RedisConnectionPool
    command_stats: Counter[str]
    socket_stats: Counter[str]
    command_types_used: Counter[bool]
    logging_handler: Any
    bot_app_info: discord.AppInfo
    old_tree_error = Callable[[discord.Interaction, discord.app_commands.AppCommandError], Coroutine[Any, Any, None]]
    # ai_agent: AiAgent

    def __init__(self):
        allowed_mentions = discord.AllowedMentions(roles=False, everyone=False, users=True)
        # intents = discord.Intents(
        #     guilds=True,
        #     members=True,
        #     bans=True,
        #     emojis=True,
        #     voice_states=True,
        #     messages=True,
        #     reactions=True,
        #     message_content=True,
        # )
        intents: discord.flags.Intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        intents.bans = True
        intents.emojis = True
        intents.voice_states = True
        intents.messages = True
        intents.reactions = True
        super().__init__(
            # command_prefix=_prefix_callable,
            command_prefix=commands.when_mentioned_or(aiosettings.prefix),
            description=DESCRIPTION,
            pm_help=None,
            help_attrs=dict(hidden=True),
            chunk_guilds_at_startup=False,
            heartbeat_timeout=150.0,
            allowed_mentions=allowed_mentions,
            intents=intents,
            enable_debug_events=True,
        )

        self.pool: RedisConnectionPool | None = None
        # ------------------------------------------------
        # from bot
        # ------------------------------------------------
        # Create a queue that we will use to store our "workload".
        self.queue: asyncio.Queue = asyncio.Queue()

        self.tasks: list[Any] = []

        self.num_workers = 3

        self.total_sleep_time = 0

        self.start_time = datetime.datetime.now()

        self.typerCtx: Dict | None = None

        #### For upscaler

        self.job_queue: Dict[Any, Any] = {}

        # self.db: RedisConnectionPool | None = None

        if aiosettings.enable_ai:
            self.ai_agent: AiAgent = AiAgent()

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.is_available():
        #     torch.set_default_tensor_type(torch.cuda.HalfTensor)

        # self.current_task = None

        self.client_id: int | str = aiosettings.discord_client_id

        # shard_id: List[datetime.datetime]
        # shows the last attempted IDENTIFYs and RESUMEs
        self.resumes: defaultdict[int, list[datetime.datetime]] = defaultdict(list)
        self.identifies: defaultdict[int, list[datetime.datetime]] = defaultdict(list)

        # in case of even further spam, add a cooldown mapping
        # for people who excessively spam commands
        self.spam_control = commands.CooldownMapping.from_cooldown(10, 12.0, commands.BucketType.user)

        # A counter to auto-ban frequent spammers
        # Triggering the rate limit 5 times in a row will auto-ban the user from the bot.
        self._auto_spam_count = Counter()

        self.channel_list = [int(x) for x in CHANNEL_ID.split(",")]

    async def setup_hook(self) -> None:
        """
        Asynchronous setup hook for initializing the bot.

        This method is called to perform asynchronous setup tasks for the bot.
        It initializes the aiohttp session, sets up guild prefixes, retrieves
        bot application information, and loads extensions.

        It also sets the intents for members and message content to True.

        Raises:
            Exception: If an extension fails to load, an exception is raised with
                       detailed error information.
        """
        self.session = aiohttp.ClientSession()
        self.prefixes: list[str] = [aiosettings.prefix]

        self.version = goob_ai.__version__
        self.guild_data: Dict[Any, Any] = {}
        self.intents.members = True
        self.intents.message_content = True

        self.bot_app_info: discord.AppInfo = await self.application_info()
        self.owner_id: int = self.bot_app_info.owner.id  # pyright: ignore[reportAttributeAccessIssue]

        for ext in extensions():
            try:
                await self.load_extension(ext)
            except Exception as ex:
                print(f"Failed to load extension {ext} - exception: {ex}")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                LOGGER.error(f"Error Class: {str(ex.__class__)}")
                output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
                LOGGER.warning(output)
                LOGGER.error(f"exc_type: {exc_type}")
                LOGGER.error(f"exc_value: {exc_value}")
                traceback.print_tb(exc_traceback)
                raise

    @property
    def owner(self) -> discord.User:
        """
        Retrieve the owner of the bot.

        This property returns the owner of the bot as a discord.User object.
        The owner information is retrieved from the bot's application info.

        Returns:
            discord.User: The owner of the bot.
        """
        return self.bot_app_info.owner  # pyright: ignore[reportAttributeAccessIssue]

    def _clear_gateway_data(self) -> None:
        """
        Clear gateway data older than one week.

        This method removes entries from the `identifies` and `resumes` dictionaries
        that are older than one week. It iterates through each shard's list of dates
        and deletes the entries that are older than the specified time frame.

        Returns:
            None
        """
        one_week_ago = discord.utils.utcnow() - datetime.timedelta(days=7)
        for shard_id, dates in self.identifies.items():
            to_remove = [index for index, dt in enumerate(dates) if dt < one_week_ago]
            for index in reversed(to_remove):
                del dates[index]

        for shard_id, dates in self.resumes.items():
            to_remove = [index for index, dt in enumerate(dates) if dt < one_week_ago]
            for index in reversed(to_remove):
                del dates[index]

    async def before_identify_hook(self, shard_id: int, *, initial: bool) -> None:
        """
        Perform actions before identifying the shard.

        This method is called before the bot identifies the shard with the Discord gateway.
        It clears old gateway data and appends the current timestamp to the identifies list
        for the given shard ID.

        Args:
            shard_id (int): The ID of the shard that is about to identify.
            initial (bool): Whether this is the initial identification of the shard.

        Returns:
            None
        """
        self._clear_gateway_data()
        self.identifies[shard_id].append(discord.utils.utcnow())
        await super().before_identify_hook(shard_id, initial=initial)

    async def on_command_error(self, ctx: Context, error: commands.CommandError) -> None:
        """
        Handle errors raised during command invocation.

        This method is called when an error is raised during the invocation of a command.
        It handles different types of command errors and sends appropriate messages to the user.

        Args:
            ctx (Context): The context in which the command was invoked.
            error (commands.CommandError): The error that was raised during command invocation.

        Returns:
            None
        """
        if isinstance(error, commands.NoPrivateMessage):
            await ctx.author.send("This command cannot be used in private messages.")
        elif isinstance(error, commands.DisabledCommand):
            await ctx.author.send("Sorry. This command is disabled and cannot be used.")
        elif isinstance(error, commands.CommandInvokeError):
            original = error.original  # pyright: ignore[reportAttributeAccessIssue]
            if not isinstance(original, discord.HTTPException):
                LOGGER.exception("In %s:", ctx.command.qualified_name, exc_info=original)
        elif isinstance(error, commands.ArgumentParsingError):
            await ctx.send(str(error))

    def get_guild_prefixes(self, guild: Optional[discord.abc.Snowflake], *, local_inject=_prefix_callable) -> list[str]:
        """
        Retrieve the command prefixes for a specific guild.

        This function generates a list of command prefixes for the bot based on the provided guild.
        If the guild is None, it returns the default prefixes. The function uses a proxy message
        to simulate a message from the guild and retrieves the prefixes using the local_inject function.

        Args:
            guild (Optional[discord.abc.Snowflake]): The guild for which to retrieve the command prefixes.
            local_inject (Callable): A callable function to inject the local context for prefix retrieval.

        Returns:
            list[str]: A list of command prefixes for the specified guild.
        """
        proxy_msg = ProxyObject(guild)
        return local_inject(self, proxy_msg)  # type: ignore  # lying

    async def query_member_named(
        self, guild: discord.Guild, argument: str, *, cache: bool = False
    ) -> Optional[discord.Member]:
        """
        Query a member by their name, name + discriminator, or nickname.

        This asynchronous function searches for a member in the specified guild
        by their name, name + discriminator (e.g., username#1234), or nickname.
        It can optionally cache the results of the query.

        Args:
            guild (discord.Guild): The guild to query the member in.
            argument (str): The name, nickname, or name + discriminator combo to check.
            cache (bool): Whether to cache the results of the query. Defaults to False.

        Returns:
            Optional[discord.Member]: The member matching the query or None if not found.
        """
        if len(argument) > 5 and argument[-5] == "#":
            username, _, discriminator = argument.rpartition("#")
            members = await guild.query_members(username, limit=100, cache=cache)
            return discord.utils.get(members, name=username, discriminator=discriminator)
        else:
            members = await guild.query_members(argument, limit=100, cache=cache)

            return discord.utils.find(lambda m: m.name == argument or m.nick == argument, members)  # pylint: disable=consider-using-in # pyright: ignore[reportAttributeAccessIssue]

    async def get_or_fetch_member(self, guild: discord.Guild, member_id: int) -> Optional[discord.Member]:
        """
        Retrieve a member from the cache or fetch from the API if not found.

        This asynchronous function attempts to retrieve a member from the cache
        in the specified guild using the provided member ID. If the member is not
        found in the cache, it fetches the member from the Discord API. The function
        handles rate limiting and returns the member if found, or None if not found.

        Args:
            guild (discord.Guild): The guild to look in.
            member_id (int): The member ID to search for.

        Returns:
            Optional[discord.Member]: The member if found, or None if not found.
        """

        member = guild.get_member(member_id)
        if member is not None:
            return member

        shard: discord.ShardInfo = self.get_shard(guild.shard_id)  # type: ignore  # will never be None
        if shard.is_ws_ratelimited():
            try:
                member = await guild.fetch_member(member_id)
            except discord.HTTPException:
                return None
            else:
                return member

        members = await guild.query_members(limit=1, user_ids=[member_id], cache=True)
        return members[0] if members else None

    async def resolve_member_ids(
        self, guild: discord.Guild, member_ids: Iterable[int]
    ) -> AsyncIterator[discord.Member]:
        """
        Bulk resolve member IDs to member instances, if possible.

        This asynchronous function attempts to resolve a list of member IDs to their corresponding
        member instances within a specified guild. Members that cannot be resolved are discarded
        from the list. The function yields the resolved members lazily using an asynchronous iterator.

        Note:
            The order of the resolved members is not guaranteed to be the same as the input order.

        Args:
            guild (discord.Guild): The guild to resolve members from.
            member_ids (Iterable[int]): An iterable of member IDs to resolve.

        Yields:
            discord.Member: The resolved members.
        """

        needs_resolution = []
        for member_id in member_ids:
            member = guild.get_member(member_id)
            if member is not None:
                yield member
            else:
                needs_resolution.append(member_id)

        total_need_resolution = len(needs_resolution)
        if total_need_resolution == 1:
            shard: discord.ShardInfo = self.get_shard(guild.shard_id)  # type: ignore  # will never be None
            if shard.is_ws_ratelimited():
                try:
                    member = await guild.fetch_member(needs_resolution[0])
                except discord.HTTPException:
                    pass
                else:
                    yield member
            else:
                members = await guild.query_members(limit=1, user_ids=needs_resolution, cache=True)
                if members:
                    yield members[0]
        elif total_need_resolution <= 100:
            # Only a single resolution call needed here
            resolved = await guild.query_members(limit=100, user_ids=needs_resolution, cache=True)
            for member in resolved:
                yield member
        else:
            # We need to chunk these in bits of 100...
            for index in range(0, total_need_resolution, 100):
                to_resolve = needs_resolution[index : index + 100]
                members = await guild.query_members(limit=100, user_ids=to_resolve, cache=True)
                for member in members:
                    yield member

    async def on_ready(self) -> None:
        """
        Handle the event when the bot is ready.

        This method is called when the bot has successfully logged in and has completed
        its initial setup. It logs the bot's user information, sets the bot's presence,
        and prints the invite link. Additionally, it preloads guild data and logs the
        logger tree structure.

        Returns:
            None
        """
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("------")
        self.invite = INVITE_LINK.format(self.user.id)
        self.guild_data = await preload_guild_data()
        print(
            f"""Logged in as {self.user}..
            Serving {len(self.users)} users in {len(self.guilds)} guilds
            Invite: {INVITE_LINK.format(self.user.id)}
        """
        )
        await self.change_presence(status=discord.Status.online, activity=discord.Game("GoobBot"))

        if not hasattr(self, "uptime"):
            self.uptime = discord.utils.utcnow()

        LOGGER.info(f"Ready: {self.user} (ID: {self.user.id})")

        LOGGER.info("LOGGING TREE:")
        await get_logger_tree_printout()

    async def on_shard_resumed(self, shard_id: int) -> None:
        """
        Handle the event when a shard resumes.

        This method is called when a shard successfully resumes its connection
        to the Discord gateway. It logs the shard ID and the timestamp of the
        resume event.

        Args:
            shard_id (int): The ID of the shard that resumed.

        Returns:
            None
        """
        LOGGER.info("Shard ID %s has resumed...", shard_id)
        self.resumes[shard_id].append(discord.utils.utcnow())

    # SOURCE: https://github.com/aronweiler/assistant/blob/a8abd34c6973c21bc248f4782f1428a810daf899/src/discord/rag_bot.py#L90
    async def process_attachments(self, message: discord.Message) -> None:
        """Process attachments in a Discord message.

        This asynchronous function processes attachments in a Discord message by downloading each attached file,
        storing it in a temporary directory, and then loading and processing the files. It sends a message to indicate
        the start of processing and handles any errors that occur during the download process.

        Args:
            message (discord.Message): The Discord message containing attachments to be processed.

        Returns:
            None
        """

        if len(message.attachments) > 0:  # pyright: ignore[reportAttributeAccessIssue]
            await message.channel.send("Processing attachments... (this may take a minute)", delete_after=30.0)  # pyright: ignore[reportAttributeAccessIssue]

            root_temp_dir = "temp/" + str(uuid.uuid4())
            uploaded_file_paths = []
            for attachment in message.attachments:  # pyright: ignore[reportAttributeAccessIssue]
                logging.debug(f"Downloading file from {attachment.url}")
                # Download the file
                file_path = os.path.join(root_temp_dir, attachment.filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Download the file from the URL
                async with aiohttp.ClientSession() as session:
                    async with session.get(attachment.url) as resp:
                        if resp.status != 200:
                            raise aiohttp.ClientException(f"Error downloading file from {attachment.url}")
                        data = await resp.read()

                        with open(file_path, "wb") as f:
                            f.write(data)

                uploaded_file_paths.append(file_path)

            # FIXME: RE ENABLE THIS SHIT 6/5/2024
            # # Process the files
            # await self.load_files(
            #     uploaded_file_paths=uploaded_file_paths,
            #     root_temp_dir=root_temp_dir,
            #     message=message,
            # )

    async def check_for_attachments(self, message: discord.Message) -> str:
        """Check a Discord message for attachments and process image URLs.

        This asynchronous function examines a Discord message for attachments,
        processes Tenor GIF URLs, downloads and processes image URLs, and modifies
        the message content based on the extracted information.

        Args:
            message (discord.Message): The Discord message to check for attachments and process.

        Returns:
            str: The updated message content with extracted information.
        """

        # Check if the message content is a URL

        message_content = message.content  # pyright: ignore[reportAttributeAccessIssue]
        url_pattern = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
        if "https://tenor.com/view/" in message_content:
            # Extract the Tenor GIF URL from the message content
            start_index = message_content.index("https://tenor.com/view/")
            end_index = message_content.find(" ", start_index)
            if end_index == -1:
                tenor_url = message_content[start_index:]
            else:
                tenor_url = message_content[start_index:end_index]
            # Split the URL on forward slashes
            parts = tenor_url.split("/")
            # Extract the relevant words from the URL
            words = parts[-1].split("-")[:-1]
            # Join the words into a sentence
            sentence = " ".join(words)
            message_content = f"{message_content} [{message.author.display_name} posts an animated {sentence} ]"
            message_content = message_content.replace(tenor_url, "")
            return message_content

        elif url_pattern.match(message_content):
            LOGGER.info(f"Message content is a URL: {message_content}")
            # Download the image from the URL and convert it to a PIL image
            response = await download_image(message_content)
            # response = requests.get(message_content)
            image = Image.open(BytesIO(response.content)).convert("RGB")  # pyright: ignore[reportAttributeAccessIssue]
        else:
            LOGGER.info(f"OTHERRRR Message content is a URL: {message_content}")
            # Download the image from the message and convert it to a PIL image
            image_url = message.attachments[0].url  # pyright: ignore[reportAttributeAccessIssue]
            # response = requests.get(image_url)
            response = await download_image(message_content)
            image = Image.open(BytesIO(response.content)).convert("RGB")  # pyright: ignore[reportAttributeAccessIssue]

        # # Generate the image caption
        # caption = self.caption_image(image)
        # message_content = f"{message_content} [{message.author.display_name} posts a picture of {caption}]"
        return message_content

    def get_attachments(
        self, message: discord.Message
    ) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]], List[str]]:
        """Retrieve attachment data from a Discord message.

        This function processes the attachments in a Discord message and converts each attachment
        to a dictionary format. It returns a tuple containing lists of dictionaries and file paths
        for further processing.

        Args:
            message (discord.Message): The Discord message containing attachments.

        Returns:
            Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]], List[str]]:
            A tuple containing:
                - A list of dictionaries with attachment data.
                - A list of local attachment file paths.
                - A list of dictionaries with local attachment data.
                - A list of media file paths.
        """

        attachment_data_list_dicts = []
        local_attachment_file_list = []
        local_attachment_data_list_dicts = []
        media_filepaths = []

        for attm in message.attachments:  # pyright: ignore[reportAttributeAccessIssue]
            data = attachment_to_dict(attm)
            attachment_data_list_dicts.append(data)

        return attachment_data_list_dicts, local_attachment_file_list, local_attachment_data_list_dicts, media_filepaths

    async def write_attachments_to_disk(self, message: discord.Message) -> None:
        """Save attachments from a Discord message to disk.

        This asynchronous function processes the attachments in a Discord message,
        saves them to a temporary directory, and logs the file paths. It also handles
        any errors that occur during the download process.

        Args:
            message (discord.Message): The Discord message containing attachments to be saved.

        Returns:
            None
        """
        ctx = await self.get_context(message)
        attachment_data_list_dicts, local_attachment_file_list, local_attachment_data_list_dicts, media_filepaths = (
            self.get_attachments(message)
        )

        # Create a temporary directory to store the attachments
        tmpdirname = "temp/" + str(uuid.uuid4())
        os.makedirs(os.path.dirname(tmpdirname), exist_ok=True)
        print("created temporary directory", tmpdirname)
        with Timer(text="\nTotal elapsed time: {:.1f}"):
            # Save each attachment to the temporary directory
            for an_attachment_dict in attachment_data_list_dicts:
                local_attachment_path = await handle_save_attachment_locally(an_attachment_dict, tmpdirname)
                local_attachment_file_list.append(local_attachment_path)

            # Create a list of dictionaries with information about the local files
            for some_file in local_attachment_file_list:
                local_data_dict = file_to_local_data_dict(some_file, tmpdirname)
                local_attachment_data_list_dicts.append(local_data_dict)
                path_to_image = file_functions.fix_path(local_data_dict["filename"])
                media_filepaths.append(path_to_image)

            print("hello")

            rich.print("media_filepaths -> ")
            rich.print(media_filepaths)

            print("standy")

            try:
                for count, media_fpaths in enumerate(media_filepaths):
                    # Compute all predictions first
                    full_path_input_file, full_path_output_file, get_timestamp = await details_from_file(
                        media_fpaths, cwd=f"{tmpdirname}"
                    )
            except Exception as ex:
                await ctx.send(embed=discord.Embed(description="Could not download story...."))
                print(ex)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                LOGGER.error(f"Error Class: {str(ex.__class__)}")
                output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
                LOGGER.warning(output)
                await ctx.send(embed=discord.Embed(description=output))
                LOGGER.error(f"exc_type: {exc_type}")
                LOGGER.error(f"exc_value: {exc_value}")
                traceback.print_tb(exc_traceback)

            # Log the directory tree of the temporary directory
            tree_list = file_functions.tree(pathlib.Path(f"{tmpdirname}"))
            rich.print("tree_list ->")
            rich.print(tree_list)

            file_to_upload_list = [f"{p}" for p in tree_list]
            LOGGER.debug(f"{type(self).__name__} -> file_to_upload_list = {file_to_upload_list}")
            rich.print(file_to_upload_list)

    def prepare_agent_input(self, message: discord.Message, user_real_name: str, surface_info: Dict) -> Dict[str, Any]:
        """Prepare the agent input from the incoming Discord message.

        This function constructs the input dictionary to be sent to the agent based on the
        provided Discord message, user's real name, and surface information. It includes
        the message content, user name, and any attachments if present.

        Args:
            message (discord.Message): The Discord message containing the user input.
            user_real_name (str): The real name of the user who sent the message.
            surface_info (Dict): The surface information related to the message.

        Returns:
            Dict[str, Any]: The input dictionary to be sent to the agent.
        """
        agent_input = {"user name": user_real_name, "message": message.content}  # pyright: ignore[reportAttributeAccessIssue]

        if len(message.attachments) > 0:  # pyright: ignore[reportAttributeAccessIssue]
            for attachment in message.attachments:  # pyright: ignore[reportAttributeAccessIssue]
                print(f"attachment -> {attachment}")  # pyright: ignore[reportAttributeAccessIssue]
                agent_input["file_name"] = attachment.filename  # pyright: ignore[reportAttributeAccessIssue]
                if attachment.content_type.startswith("image/"):
                    agent_input["image_url"] = attachment.url  # pyright: ignore[reportAttributeAccessIssue]

        agent_input["surface_info"] = surface_info

        return agent_input

    def get_session_id(self, message: discord.Message) -> str:
        """Generate a session ID for the given message.

        This function generates a session ID based on the message context.
        The session ID is used as a key for the history session and as an identifier for logs.

        Args:
            message (discord.Message): The message or event dictionary.

        Returns:
            str: The generated session ID.

        Notes:
            - If the message is a direct message (DM), the session ID is based on the user ID.
            - If the message is from a guild (server) channel, the session ID is based on the channel ID.
        """
        is_dm: bool = str(message.channel.type) == "private"  # pyright: ignore[reportAttributeAccessIssue]
        user_id: int = message.author.id  # pyright: ignore[reportAttributeAccessIssue]

        if is_dm:
            return f"discord_{user_id}"
        else:
            return f"discord_{message.channel.id}"  # pyright: ignore[reportAttributeAccessIssue]

        # TODO: ENABLE THIS AND THREAD HANDLING
        # channel_id = message.channel.id

        # thread_ts = msg.get("thread_ts")
        # if thread_ts is None:
        #     thread_ts = msg.get("ts")
        # if thread_ts is None and "message" in msg:
        #     thread_ts = msg.get("message").get("thread_ts", None)
        #     if thread_ts is None:
        #         thread_ts = msg.get("message").get("ts", None)
        # return f"slack_{channel_id}_{thread_ts}"

    async def handle_dm_from_user(self, message: discord.Message) -> bool:
        """
        Handle a direct message (DM) from a user.

        This asynchronous function processes a direct message (DM) received from a user.
        It prepares the surface information, constructs the agent input, and processes
        the user task using the AI agent. The function sends a temporary message to the
        user indicating that the request is being processed and then sends the agent's
        response in multiple messages if necessary.

        Args:
            message (discord.Message): The Discord message object representing the direct message.

        Returns:
            bool: True if the message was successfully processed, False otherwise.

        Example:
            >>> await bot.handle_dm_from_user(message)
        """
        ctx = await self.get_context(message)
        # Process DMs
        LOGGER.info("Processing direct message")

        # For direct messages, prepare the surface info
        surface_info = SurfaceInfo(surface=SurfaceType.DISCORD, type="direct_message", source="direct_message")

        # Convert surface_info to dictionary using the utility function
        surface_info_dict = surface_info.__dict__

        user_name = message.author.name  # pyright: ignore[reportAttributeAccessIssue]
        agent_input = self.prepare_agent_input(message, user_name, surface_info_dict)
        session_id = self.get_session_id(message)
        LOGGER.info(f"session_id: {session_id} Agent input: {json.dumps(agent_input)}")

        temp_message = (
            "Processing your request, please wait... (this could take up to 1min, depending on the response size)"
        )
        # thread_ts is None for direct messages, so we use the ts of the original message
        orig_msg = await ctx.send(
            # embed=discord.Embed(description=temp_message),
            content=temp_message,
            delete_after=30.0,
        )
        # response = client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=temp_message)
        # response_ts = response["ts"]
        # For direct messages, remember chat based on the user:

        agent_response_text = self.ai_agent.process_user_task(session_id, str(agent_input))

        # Update the slack response with the agent response
        # client.chat_update(channel=channel_id, ts=response_ts, text=agent_response_text)

        # Sometimes the response can be over 2000 characters, so we need to split it
        # into multiple messages, and send them one at a time
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", ".", "?", "!"],
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len,
        )

        responses = text_splitter.split_text(agent_response_text)

        for rsp in responses:
            await message.channel.send(rsp)  # pyright: ignore[reportAttributeAccessIssue]

        # await orig_msg.edit(content=agent_response_text)
        LOGGER.info(f"session_id: {session_id} Agent response: {json.dumps(agent_response_text)}")

        # Evaluate the response against the question
        eval_result = Evaluator().evaluate_prediction(agent_input, agent_response_text)
        # Log the evaluation result
        LOGGER.info(f"session_id: {session_id} Evaluation Result: {eval_result}")
        return True

    async def get_context(self, origin: Union[discord.Interaction, discord.Message], /, *, cls=Context) -> Context:
        """Retrieve the context for a Discord interaction or message.

        This asynchronous method retrieves the context for a Discord interaction or message
        and returns a Context object. It calls the superclass method to get the context
        based on the provided origin and class type.

        Args:
            origin (Union[discord.Interaction, discord.Message]): The Discord interaction or message to get the context from.
            cls (Context): The class type for the context object.

        Returns:
            Context: The context object retrieved for the provided origin.
        """

        return await super().get_context(origin, cls=cls)

    async def process_commands(self, message: discord.Message) -> None:
        """Process commands based on the received Discord message.

        This asynchronous function processes commands based on the provided Discord message.
        It retrieves the context for the message, logs information, and then invokes the command handling.
        It includes commented-out sections for potential future functionality like spam control and blacklisting.

        Args:
            message (discord.Message): The Discord message to process commands from.

        Returns:
            None
        """

        ctx = await self.get_context(message)

        # import bpdb
        # bpdb.set_trace()

        LOGGER.info(f"ctx = {ctx}")

        # TODO: reenable this if you just want to verify that discord is getting messages
        # if ctx.command is None:
        #     return

        # >>> debugger.dump_magic(ctx)
        # obj._get_channel = <bound method Context._get_channel of <Context>>
        # obj._is_protocol = False
        # obj._state = <discord.state.ConnectionState object at 0x12e9079a0>
        # obj.args = []
        # obj.author = bossjones
        # obj.bot = <goob_ai.goob_bot.AsyncGoobBot object at 0x12e9077c0>
        # obj.bot_permissions = <Permissions value=70508331125824>
        # obj.channel = Direct Message with Unknown User
        # obj.clean_prefix =
        # obj.cog = None
        # obj.command = None
        # obj.command_failed = False
        # obj.current_argument = None
        # obj.current_parameter = None
        # obj.db = ConnectionPool<Connection<host=localhost,port=7600,db=0>>
        # obj.defer = <bound method Context.defer of <Context>>
        # obj.disambiguate = <bound method Context.disambiguate of <Context>>
        # obj.entry_to_code = <bound method Context.entry_to_code of <Context>>
        # obj.fetch_message = <bound method Messageable.fetch_message of <Context>>
        # obj.filesize_limit = 26214400
        # obj.from_interaction = <bound method Context.from_interaction of <class 'goob_ai.utils.context.Context'>>
        # obj.guild = None
        # obj.history = <bound method Messageable.history of <Context>>
        # obj.indented_entry_to_code = <bound method Context.indented_entry_to_code of <Context>>
        # obj.interaction = None
        # obj.invoke = <bound method Context.invoke of <Context>>
        # obj.invoked_parents = []
        # obj.invoked_subcommand = None
        # obj.invoked_with = None
        # obj.kwargs = {}
        # obj.me = GoobAI#5251
        # obj.message = <Message id=1248071351072194641 channel=<DMChannel id=1237526936201334804 recipient=None> type=<MessageType.default: 0> author=<User id=304597333860155393 name='bossjones' global_name='bossjones'
        # bot=False> flags=<MessageFlags value=0>>
        # obj.permissions = <Permissions value=70508331125824>
        # obj.pins = <bound method Messageable.pins of <Context>>
        # obj.pool = ConnectionPool<Connection<host=localhost,port=7600,db=0>>
        # obj.prefix = None
        # obj.prompt = <bound method Context.prompt of <Context>>
        # obj.reinvoke = <bound method Context.reinvoke of <Context>>
        # obj.replied_message = None
        # obj.replied_reference = None
        # obj.reply = <bound method Context.reply of <Context>>
        # obj.safe_send = <bound method Context.safe_send of <Context>>
        # obj.send = <bound method Context.send of <Context>>
        # obj.send_help = <bound method Context.send_help of <Context>>
        # obj.session = <aiohttp.client.ClientSession object at 0x12e907760>
        # obj.show_help = <bound method Context.show_help of <Context>>
        # obj.subcommand_passed = None
        # obj.tick = <bound method Context.tick of <Context>>
        # obj.typing = <bound method Context.typing of <Context>>
        # obj.valid = False
        # obj.view = <StringView pos: 0 prev: 0 end: 12 eof: False>
        # obj.voice_client = None
        # >>>

        # if ctx.author.id in self.blacklist:
        #     return

        # if ctx.guild is not None and ctx.guild.id in self.blacklist:
        #     return

        # bucket = self.spam_control.get_bucket(message)
        # current = message.created_at.timestamp()
        # retry_after = bucket and bucket.update_rate_limit(current)
        # author_id = message.author.id
        # if retry_after and author_id != self.owner_id:
        #     self._auto_spam_count[author_id] += 1
        #     if self._auto_spam_count[author_id] >= 5:
        #         # await self.add_to_blacklist(author_id)
        #         del self._auto_spam_count[author_id]
        #         await self.log_spammer(ctx, message, retry_after, autoblock=True)
        #     else:
        #         await self.log_spammer(ctx, message, retry_after)
        #     return
        # else:
        #     self._auto_spam_count.pop(author_id, None)

        # its a dm
        if str(message.channel.type) == "private":  # pyright: ignore[reportAttributeAccessIssue]
            await self.handle_dm_from_user(message)

        await self.invoke(ctx)

    async def handle_user_task(self, message: discord.Message) -> JSONResponse:
        """Handle a user task received through a Discord message.

        This asynchronous function processes a user task received as a Discord message.
        It determines the surface type of the message (DM or channel), creates a SurfaceInfo
        instance, prepares the agent input, and processes the user task using the AI agent.
        It returns a JSON response with the agent's text response or raises an exception if
        processing fails.

        Args:
            message (discord.Message): The Discord message containing the user task.

        Returns:
            JSONResponse: A JSON response containing the agent's text response with a status code of 200 if successful.

        Raises:
            Exception: If an error occurs during the processing of the user task.
        """

        # Optional surface information
        surface = "discord"
        if isinstance(message.channel, discord.DMChannel):  # pyright: ignore[reportAttributeAccessIssue]
            LOGGER.debug("Message is from a DM channel")
            surface_type = "dm"
            source = "dm"
        elif isinstance(message.channel, discord.TextChannel):  # pyright: ignore[reportAttributeAccessIssue]
            LOGGER.debug("Message is from a text channel")
            surface_type = "channel"
            source = message.channel.name  # pyright: ignore[reportAttributeAccessIssue]

        # Convert surface to SurfaceType enum, default to UNKNOWN if not found
        surface_enum = SurfaceType[surface.upper()] if surface else SurfaceType.UNKNOWN

        # Create an instance of SurfaceInfo
        surface_info = SurfaceInfo(surface=surface_enum, type=surface_type, source=source)

        agent_input = {"user name": message.author, "message": message.content, "surface_info": surface_info.__dict__}  # pyright: ignore[reportAttributeAccessIssue]

        # Modify the call to process_user_task to pass agent_input
        # if not is_streaming:
        try:
            agent_response_text = self.ai_agent.process_user_task(REQUEST_ID_CONTEXTVAR.get(), str(agent_input))
            return JSONResponse(content={"response": agent_response_text}, status_code=200)
        except Exception as ex:
            LOGGER.exception(f"Failed to process user task: {ex}")
            print(f"Failed to load extension {ex} - exception: {ex}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            LOGGER.error(f"Error Class: {str(ex.__class__)}")
            output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
            LOGGER.warning(output)
            LOGGER.error(f"exc_type: {exc_type}")
            LOGGER.error(f"exc_value: {exc_value}")
            traceback.print_tb(exc_traceback)
            raise
        # except Exception as e:
        #     LOGGER.exception(f"Failed to process user task: {e}")
        #     return HTTPException(status_code=500, detail=str(e))

    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming messages and process commands.

        This method is called whenever a message is received. It logs the message details,
        processes any attachments, and then processes the message content as a command if applicable.

        Args:
            message (discord.Message): The message object received from Discord.

        Returns:
            None
        """
        LOGGER.info(f"message = {message}")
        LOGGER.info(f"ITS THIS ONE BOSS")

        LOGGER.info(f"You are in function: {CURRENTFUNCNAME()}")
        LOGGER.info(f"This function's caller was: {CURRENTFUNCNAME(1)}")

        # TODO: This is where all the AI logic is going to go
        LOGGER.info(f"Thread message to process - {message.author}: {message.content[:50]}")  # pyright: ignore[reportAttributeAccessIssue]
        if message.author.bot:
            return

        # Handle attachments first
        await self.process_attachments(message)
        if message.content.strip() != "":  # pyright: ignore[reportAttributeAccessIssue]
            # NOTE: dptest doesn't like this, disable so we can keep tests # async with message.channel.typing():  # pyright: ignore[reportAttributeAccessIssue]
            # Send everything to AI bot
            await self.process_commands(message)

    async def close(self) -> None:
        await super().close()
        await self.session.close()

    async def start(self) -> None:
        await super().start(aiosettings.discord_token, reconnect=True)

    async def my_background_task(self) -> None:
        """_summary_"""
        await self.wait_until_ready()
        counter = 0
        # TEMPCHANGE: # channel = self.get_channel(DISCORD_GENERAL_CHANNEL)  # channel ID goes here
        channel = self.get_channel(aiosettings.discord_general_channel)  # channel ID goes here
        while not self.is_closed():
            counter += 1
            await channel.send(counter)  # pyright: ignore[reportAttributeAccessIssue]
            await asyncio.sleep(60)  # task runs every 60 seconds

    async def on_worker_monitor(self) -> None:
        await self.wait_until_ready()
        counter = 0
        # channel = self.get_channel(DISCORD_GENERAL_CHANNEL)  # channel ID goes here
        while not self.is_closed():
            counter += 1
            # await channel.send(counter)
            print(f" self.tasks = {self.tasks}")
            print(f" len(self.tasks) = {len(self.tasks)}")
            await asyncio.sleep(10)  # task runs every 60 seconds'

    # def blah_blah(self) -> None:


# SOURCE: https://github.com/darren-rose/DiscordDocChatBot/blob/63a2f25d2cb8aaace6c1a0af97d48f664588e94e/main.py#L28
# TODO: maybe enable this
async def send_long_message(channel: Any, message: discord.Message, max_length: int = 2000) -> None:
    """
    Summary:
    Send a long message by splitting it into chunks and sending each chunk.

    Explanation:
    This asynchronous function takes a message and splits it into chunks of maximum length 'max_length'. It then sends each chunk as a separate message to the specified channel.

    Args:
    - channel (Any): The channel to send the message chunks to.
    - message (discord.Message): The message to be split into chunks and sent.

    Returns:
    - None
    """

    chunks = [message[i : i + max_length] for i in range(0, len(message), max_length)]
    for chunk in chunks:
        await channel.send(chunk)
