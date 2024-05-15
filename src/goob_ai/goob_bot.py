"""goob_ai.goob_bot"""

# pyright: reportImportCycles=false
from __future__ import annotations

import asyncio
import datetime
import logging
import os
import pathlib
import sys
import time
import traceback

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

from codetiming import Timer
from discord.ext import commands
from logging_tree import printout
from loguru import logger as LOGGER
from redis.asyncio import ConnectionPool as RedisConnectionPool

import goob_ai

from goob_ai import db, helpers, shell, utils
from goob_ai.agent import AiAgent
from goob_ai.aio_settings import aiosettings
from goob_ai.bot_logger import generate_tree, get_lm_from_tree, get_logger
from goob_ai.constants import CHANNEL_ID, INPUT_CLASSIFICATION_NOT_A_QUESTION, INPUT_CLASSIFICATION_NOT_FOR_ME
from goob_ai.factories import guild_factory
from goob_ai.user_input_enrichment import UserInputEnrichment
from goob_ai.utils.context import Context


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
    lm = get_lm_from_tree(rootm, logger_name)
    return lm


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
        base.append("!")
        base.append("?")
    else:
        base.extend(bot.prefixes.get(msg.guild.id, ["?", "!"]))  # pyright: ignore[reportAttributeAccessIssue]
    return base


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
        if not members:
            return None
        return members[0]

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

    async def get_context(self, origin: Union[discord.Interaction, discord.Message], /, *, cls=Context) -> Context:
        return await super().get_context(origin, cls=cls)

    async def process_commands(self, message: discord.Message):
        ctx = await self.get_context(message)

        if ctx.command is None:
            return

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

        await self.invoke(ctx)

    async def on_message(self, message: discord.Message) -> None:
        LOGGER.info(f"message = {message}")

        # TODO: This is where all the AI logic is going to go
        LOGGER.info(f"Thread message to process - {message.author}: {message.content[:50]}")  # pyright: ignore[reportAttributeAccessIssue]
        if message.author.bot:
            return
        await self.process_commands(message)

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
