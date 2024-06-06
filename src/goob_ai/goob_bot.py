"""goob_ai.goob_bot"""

# pyright: reportImportCycles=false
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import pathlib
import sys
import time
import traceback
import typing

from collections import Counter, defaultdict
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
    TypeVar,
    Union,
    cast,
)

import aiohttp
import discord
import rich

from codetiming import Timer
from discord.ext import commands
from logging_tree import printout
from loguru import logger as LOGGER
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
from goob_ai.utils.context import Context
from goob_ai.utils.misc import CURRENTFUNCNAME


LOGGER.add(sys.stderr, level="DEBUG")

DESCRIPTION = """An example bot to showcase the discord.ext.commands extension
module.

There are a number of utility commands being showcased here."""

HERE = os.path.dirname(__file__)

INVITE_LINK = "https://discordapp.com/api/oauth2/authorize?client_id={}&scope=bot&permissions=0"

# LOGGER = get_logger(__name__, provider="Bot", level=logging.DEBUG)


HOME_PATH = os.environ.get("HOME")

COMMAND_RUNNER = {"dl_thumb": shell.run_coroutine_subprocess}
from goob_ai.utils import async_


@async_.to_async
def get_logger_tree_printout() -> None:
    printout()


# SOURCE: https://realpython.com/how-to-make-a-discord-bot-python/#responding-to-messages
def dump_logger_tree():
    rootm = generate_tree()
    LOGGER.debug(rootm)


def dump_logger(logger_name: str):
    LOGGER.debug(f"getting logger {logger_name}")
    rootm = generate_tree()
    return get_lm_from_tree(rootm, logger_name)


def filter_empty_string(a_list: List[str]) -> List[str]:
    """_summary_

    Args:
        a_list (List[str]): _description_

    Returns:
        List[str]: _description_
    """
    # filter out empty strings
    filter_object = filter(lambda x: x != "", a_list)

    return list(filter_object)


# SOURCE: https://docs.python.org/3/library/asyncio-queue.html
async def worker(name: str, queue: asyncio.Queue) -> NoReturn:
    """_summary_

    Args:
        name (str): _description_
        queue (asyncio.Queue): _description_

    Returns:
        NoReturn: _description_
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
async def co_task(name: str, queue: asyncio.Queue):
    """_summary_

    Args:
        name (str): _description_
        queue (asyncio.Queue): _description_
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
async def get_prefix(_bot: AsyncGoobBot, message: discord.Message):
    """_summary_

    Args:
        _bot (AsyncGoobBot): _description_
        message (discord.message.Message): _description_

    Returns:
        _type_: _description_
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
async def preload_guild_data():
    """_summary_

    Returns:
        _type_: _description_
    """
    LOGGER.info("preload_guild_data ... ")
    guilds = [guild_factory.Guild()]
    return {guild.id: {"prefix": guild.prefix} for guild in guilds}


def extensions():
    """_summary_

    Yields:
        _type_: _description_
    """
    module_dir = pathlib.Path(HERE)
    files = pathlib.Path(module_dir.stem, "cogs").rglob("*.py")
    for file in files:
        LOGGER.debug(f"exension = {file.as_posix()[:-3].replace('/', '.')}")
        yield file.as_posix()[:-3].replace("/", ".")


def _prefix_callable(bot: AsyncGoobBot, msg: discord.Message):
    user_id = bot.user.id
    base = [f"<@!{user_id}> ", f"<@{user_id}> "]
    if msg.guild is None:  # pyright: ignore[reportAttributeAccessIssue]
        base.extend(("!", "?"))
    else:
        base.extend(bot.prefixes.get(msg.guild.id, ["?", "!"]))  # pyright: ignore[reportAttributeAccessIssue]
    return base


async def details_from_file(path_to_media_from_cli: str, cwd: typing.Union[str, None] = None):
    """Take a file path and return the input and output file paths and the timestamp of the input file.

    Args:
        path_to_media_from_cli (str): _description_

    Returns:
        _type_: _description_
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


def unlink_orig_file(a_filepath: str):
    """_summary_

    Args:
        a_filepath (str): _description_

    Returns:
        _type_: _description_
    """
    # for orig_to_rm in media_filepaths:
    rich.print(f"deleting ... {a_filepath}")
    os.unlink(f"{a_filepath}")
    return a_filepath


# https://github.com/discord-math/bot/blob/babb41b71a68b4b099684b3e1ed583f84083f971/plugins/log.py#L63
def path_for(attm: discord.Attachment, basedir: str = "./") -> pathlib.Path:
    p = pathlib.Path(basedir, str(attm.filename))  # pyright: ignore[reportAttributeAccessIssue]
    LOGGER.debug(f"path_for: p -> {p}")
    return p


# https://github.com/discord-math/bot/blob/babb41b71a68b4b099684b3e1ed583f84083f971/plugins/log.py#L63
async def save_attachment(attm: discord.Attachment, basedir: str = "./") -> None:
    path = path_for(attm, basedir=basedir)
    LOGGER.debug(f"save_attachment: path -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ret_code = await attm.save(path, use_cached=True)
        await asyncio.sleep(5)
    except discord.HTTPException:
        await attm.save(path)


# TODO: Remove this when we eventually upgrade to 2.0 discord.py
def attachment_to_dict(attm: discord.Attachment):
    """Converts a discord.Attachment object to a dictionary.

    Args:
        attm (discord.Attachment): _description_

    Returns:
        _type_: _description_
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
        self.session = aiohttp.ClientSession()
        # guild_id: list
        # self.prefixes: Config[list[str]] = Config('prefixes.json')
        self.prefixes: list[str] = [aiosettings.prefix]

        # guild_id and user_id mapped to True
        # these are users and guilds globally blacklisted
        # from using the bot
        # self.blacklist: Config[bool] = Config('blacklist.json')

        self.version = goob_ai.__version__
        self.guild_data: Dict[Any, Any] = {}
        self.intents.members = True
        self.intents.message_content = True

        # if aiosettings.enable_redis:
        #     self.db = db.init_worker_redis()

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
        return self.bot_app_info.owner  # pyright: ignore[reportAttributeAccessIssue]

    def _clear_gateway_data(self) -> None:
        one_week_ago = discord.utils.utcnow() - datetime.timedelta(days=7)
        for shard_id, dates in self.identifies.items():
            to_remove = [index for index, dt in enumerate(dates) if dt < one_week_ago]
            for index in reversed(to_remove):
                del dates[index]

        for shard_id, dates in self.resumes.items():
            to_remove = [index for index, dt in enumerate(dates) if dt < one_week_ago]
            for index in reversed(to_remove):
                del dates[index]

    async def before_identify_hook(self, shard_id: int, *, initial: bool):
        self._clear_gateway_data()
        self.identifies[shard_id].append(discord.utils.utcnow())
        await super().before_identify_hook(shard_id, initial=initial)

    async def on_command_error(self, ctx: Context, error: commands.CommandError) -> None:
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
        proxy_msg = ProxyObject(guild)
        return local_inject(self, proxy_msg)  # type: ignore  # lying

    # def get_raw_guild_prefixes(self, guild_id: int) -> list[str]:
    #     return self.prefixes.get(guild_id, ["?", "!"])

    # async def set_guild_prefixes(self, guild: discord.abc.Snowflake, prefixes: list[str]) -> None:
    #     if len(prefixes) == 0:
    #         await self.prefixes.put(guild.id, [])
    #     elif len(prefixes) > 10:
    #         raise RuntimeError("Cannot have more than 10 custom prefixes.")
    #     else:
    #         await self.prefixes.put(guild.id, sorted(set(prefixes), reverse=True))

    # async def add_to_blacklist(self, object_id: int):
    #     await self.blacklist.put(object_id, True)

    # async def remove_from_blacklist(self, object_id: int):
    #     try:
    #         await self.blacklist.remove(object_id)
    #     except KeyError:
    #         pass

    async def query_member_named(
        self, guild: discord.Guild, argument: str, *, cache: bool = False
    ) -> Optional[discord.Member]:
        """Queries a member by their name, name + discrim, or nickname.

        Parameters
        ------------
        guild: Guild
            The guild to query the member in.
        argument: str
            The name, nickname, or name + discrim combo to check.
        cache: bool
            Whether to cache the results of the query.

        Returns
        ---------
        Optional[Member]
            The member matching the query or None if not found.
        """
        if len(argument) > 5 and argument[-5] == "#":
            username, _, discriminator = argument.rpartition("#")
            members = await guild.query_members(username, limit=100, cache=cache)
            return discord.utils.get(members, name=username, discriminator=discriminator)
        else:
            members = await guild.query_members(argument, limit=100, cache=cache)

            return discord.utils.find(lambda m: m.name == argument or m.nick == argument, members)  # pylint: disable=consider-using-in # pyright: ignore[reportAttributeAccessIssue]

    async def get_or_fetch_member(self, guild: discord.Guild, member_id: int) -> Optional[discord.Member]:
        """Looks up a member in cache or fetches if not found.

        Parameters
        -----------
        guild: Guild
            The guild to look in.
        member_id: int
            The member ID to search for.

        Returns
        ---------
        Optional[Member]
            The member or None if not found.
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
        """Bulk resolves member IDs to member instances, if possible.

        Members that can't be resolved are discarded from the list.

        This is done lazily using an asynchronous iterator.

        Note that the order of the resolved members is not the same as the input.

        Parameters
        -----------
        guild: Guild
            The guild to resolve from.
        member_ids: Iterable[int]
            An iterable of member IDs.

        Yields
        --------
        Member
            The resolved members.
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

    async def on_ready(self):
        """Event is called when the bot has finished logging in and setting things up"""
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

    async def on_shard_resumed(self, shard_id: int):
        LOGGER.info("Shard ID %s has resumed...", shard_id)
        self.resumes[shard_id].append(discord.utils.utcnow())

    # @discord.utils.cached_property # pyright: ignore[reportAttributeAccessIssue]
    # def stats_webhook(self) -> discord.Webhook:
    #     wh_id, wh_token = self.aiosettings.stat_webhook
    #     hook = discord.Webhook.partial(id=wh_id, token=wh_token, session=self.session)
    #     return hook

    # async def log_spammer(self, ctx: Context, message: discord.Message, retry_after: float, *, autoblock: bool = False):
    #     guild_name = getattr(ctx.guild, "name", "No Guild (DMs)")
    #     guild_id = getattr(ctx.guild, "id", None)
    #     fmt = "User %s (ID %s) in guild %r (ID %s) spamming, retry_after: %.2fs"
    #     LOGGER.warning(fmt, message.author, message.author.id, guild_name, guild_id, retry_after)
    #     if not autoblock:
    #         return

    #     wh = self.stats_webhook
    #     embed = discord.Embed(title="Auto-blocked Member", colour=0xDDA453)
    #     embed.add_field(name="Member", value=f"{message.author} (ID: {message.author.id})", inline=False)
    #     embed.add_field(name="Guild Info", value=f"{guild_name} (ID: {guild_id})", inline=False)
    #     embed.add_field(name="Channel Info", value=f"{message.channel} (ID: {message.channel.id}", inline=False) # pyright: ignore[reportAttributeAccessIssue]
    #     embed.timestamp = discord.utils.utcnow()
    #     return await wh.send(embed=embed)

    def get_attachments(self, message: discord.Message):
        attachment_data_list_dicts = []
        local_attachment_file_list = []
        local_attachment_data_list_dicts = []
        media_filepaths = []

        for attm in message.attachments:  # pyright: ignore[reportAttributeAccessIssue]
            data = attachment_to_dict(attm)
            attachment_data_list_dicts.append(data)

        return attachment_data_list_dicts

    def prepare_agent_input(self, message: discord.Message, user_real_name: str, surface_info: Dict) -> Dict[str, Any]:
        """
        Prepares the agent input from the incoming Slack event.
        params:
        - event: the incoming Slack event
        - user_real_name: the real name of the user who sent the event
        - surface_info: the surface information
        returns:
        - Dict[str, Any]: the input we want to send to the agent
        """
        agent_input = {"user name": user_real_name, "message": message.content}  # pyright: ignore[reportAttributeAccessIssue]

        # TODO: ENABLE ATTACHMENT HANDLING
        # TODO: ENABLE ATTACHMENT HANDLING
        # TODO: ENABLE ATTACHMENT HANDLING
        # TODO: ENABLE ATTACHMENT HANDLING
        # if message.attachments:
        # if "files" in event:
        #     for file_info in event["files"]:
        #         agent_input["file_name"] = file_info["name"]
        #         # Handle images
        #         if file_info["mimetype"].startswith("image/"):
        #             agent_input["image_url"] = file_info["url_private"]
        #         # Handle text-based files, PDFs, and .txt files identified either by MIME type or name pattern
        #         # .txt files sometimes show up as application/octet-stream MIME type
        #         elif (
        #             file_info["mimetype"].startswith("text/")
        #             or file_info["mimetype"] == "application/pdf"
        #             or file_info["mimetype"]
        #             == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        #             or (file_info["mimetype"] == "application/octet-stream" and file_info["name"].endswith(".txt"))
        #         ):
        #             agent_input["file_url"] = file_info["url_private"]

        # Add surface information to the agent input
        agent_input["surface_info"] = surface_info

        return agent_input

    def get_session_id(self, message: discord.Message) -> str:
        """
        Get the session ID for the message.
        Used as a key for the history session, and as an identifier for logs.
        :param msg: the message or event dictionary.
        :param is_dm: whether the message is a direct message.
        :return: the session ID
        """
        if str(message.channel.type) == "private":  # pyright: ignore[reportAttributeAccessIssue]
            is_dm = True
        else:
            is_dm = False

        user_id = message.author.id  # pyright: ignore[reportAttributeAccessIssue]

        if is_dm:
            return f"discord_{user_id}"

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

    async def handle_dm_from_user(self, message: discord.Message):
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
            embed=discord.Embed(description=temp_message),
            # delete_after=30.0,
        )
        # response = client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=temp_message)
        # response_ts = response["ts"]
        # For direct messages, remember chat based on the user:

        agent_response_text = self.ai_agent.process_user_task(session_id, str(agent_input))

        # Update the slack response with the agent response
        # client.chat_update(channel=channel_id, ts=response_ts, text=agent_response_text)

        # await orig_msg.edit(content=agent_response_text)
        await orig_msg.edit(embed=discord.Embed(description=agent_response_text))
        LOGGER.info(f"session_id: {session_id} Agent response: {json.dumps(agent_response_text)}")

        # Evaluate the response against the question
        eval_result = Evaluator().evaluate_prediction(agent_input, agent_response_text)
        # Log the evaluation result
        LOGGER.info(f"session_id: {session_id} Evaluation Result: {eval_result}")
        return True

    async def get_context(self, origin: Union[discord.Interaction, discord.Message], /, *, cls=Context) -> Context:
        """
        Summary:
        Get the context for a Discord interaction or message.

        Explanation:
        This asynchronous method retrieves the context for a Discord interaction or message and returns a Context object. It calls the superclass method to get the context based on the provided origin and class type.

        Args:
        - self: The instance of the class.
        - origin (Union[discord.Interaction, discord.Message]): The Discord interaction or message to get the context from.
        - cls (Context): The class type for the context object.

        Returns:
        - Context: The context object retrieved for the provided origin.
        """

        return await super().get_context(origin, cls=cls)

    async def process_commands(self, message: discord.Message):
        """
        Summary:
        Process commands based on the received Discord message.

        Explanation:
        This asynchronous function processes commands based on the provided Discord message. It retrieves the context for the message, logs information, and then invokes the command handling. It includes commented-out sections for potential future functionality like spam control and blacklisting.

        Args:
        - self: The instance of the class.
        - message (discord.Message): The Discord message to process commands from.

        Returns:
        - None
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

    async def handle_user_task(self, message: discord.Message):
        """
        Summary:
        Handle a user task received through a Discord message.

        Explanation:
        This asynchronous function handles a user task received as a Discord message. It determines the surface type of the message (DM or channel), creates a SurfaceInfo instance, prepares the agent input, and processes the user task using the AI agent. It returns a JSON response with the agent's text response or raises an exception if processing fails.

        Args:
        - self: The instance of the class.
        - message (discord.Message): The Discord message containing the user task.

        Returns:
        - JSONResponse: A JSON response containing the agent's text response with a status code of 200 if successful.
        - Raises: Any exception encountered during the user task processing.
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
        LOGGER.info(f"message = {message}")
        LOGGER.info(f"ITS THIS ONE BOSS")

        LOGGER.info(f"You are in function: {CURRENTFUNCNAME()}")
        LOGGER.info(f"This function's caller was: {CURRENTFUNCNAME(1)}")

        # TODO: This is where all the AI logic is going to go
        LOGGER.info(f"Thread message to process - {message.author}: {message.content[:50]}")  # pyright: ignore[reportAttributeAccessIssue]
        if message.author.bot:
            return
        await self.process_commands(message)

        # import bpdb
        # bpdb.set_trace()

    # async def on_guild_join(self, guild: discord.Guild) -> None:
    #     if guild.id in self.blacklist:
    #         await guild.leave()

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
            await asyncio.sleep(10)  # task runs every 60 seconds

    async def setup_workers(self) -> None:
        await self.wait_until_ready()

        # Create three worker tasks to process the queue concurrently.

        for i in range(self.num_workers):
            task = asyncio.create_task(worker(f"worker-{i}", self.queue))
            self.tasks.append(task)

        # Wait until the queue is fully processed.
        started_at = time.monotonic()
        await self.queue.join()
        total_slept_for = time.monotonic() - started_at

        # Cancel our worker tasks.
        for task in self.tasks:
            task.cancel()
        # Wait until all worker tasks are cancelled.
        await asyncio.gather(*self.tasks, return_exceptions=True)

        print("====")
        print(f"3 workers slept in parallel for {total_slept_for:.2f} seconds")

    async def setup_co_tasks(self) -> None:
        await self.wait_until_ready()

        # Create three worker tasks to process the queue concurrently.

        for i in range(self.num_workers):
            task = asyncio.create_task(co_task(f"worker-{i}", self.queue))
            self.tasks.append(task)

        # Wait until the queue is fully processed.
        started_at = time.monotonic()
        await self.queue.join()
        total_slept_for = time.monotonic() - started_at

        # Cancel our worker tasks.
        for task in self.tasks:
            task.cancel()
        # Wait until all worker tasks are cancelled.
        await asyncio.gather(*self.tasks, return_exceptions=True)

        print("====")
        print(f"3 workers slept in parallel for {total_slept_for:.2f} seconds")

    # TODO: Need to get this working 5/5/2024
    def input_classifier(self, event) -> bool:
        """
        Determines whether the bot should respond to a message in a channel or group.

        :param event: the incoming Slack event
        :return: True if the bot should respond, False otherwise
        """
        LOGGER.info(f"event = {event}")
        LOGGER.info(f"type(event) = {type(event)}")
        try:
            classification = UserInputEnrichment().input_classifier_tool(event.get("text", ""))

            # Explicitly not respond to "Not a question" or "Not for me"
            if classification.get("classification") in [
                INPUT_CLASSIFICATION_NOT_A_QUESTION,
                INPUT_CLASSIFICATION_NOT_FOR_ME,
            ]:
                return False
        except Exception as e:
            # Log the error but choose to respond since the classification is uncertain
            LOGGER.error(f"Error during classification, but choosing to respond: {e}")

            # Default behavior is to respond unless it's explicitly classified as "Not a question" or "Not for me"
            return True

    # @property
    # def config(self):
    #     return __import__('config')

    # @property
    # def reminder(self) -> Optional[Reminder]:
    #     return self.get_cog('Reminder')  # type: ignore

    # @property
    # def config_cog(self) -> Optional[ConfigCog]:
    #     return self.get_cog('Config')  # type: ignore
