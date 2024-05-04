#!/usr/bin/env python
"""goob_ai.bot"""

from __future__ import annotations

# SOURCE: https://realpython.com/how-to-make-a-discord-bot-python/#responding-to-messages

import asyncio
import datetime
import logging
import os
import os.path
import pathlib
import re
import sys
import time
import traceback
from typing import List, NoReturn

from aiodebug import log_slow_callbacks as aiodebug_log_slow_callbacks
import aiomonitor
from codetiming import Timer
import discord
from discord.ext import commands
import torch  # type: ignore

import goob_ai
from goob_ai import db, helpers, shell, utils
from goob_ai.aio_settings import aiosettings
from goob_ai.bot_logger import get_logger, intercept_all_loggers
from goob_ai.factories import guild_factory

# from goob_ai.web import GoobBotMetricsApi

HERE = os.path.dirname(__file__)

LOGGER = get_logger(__name__, provider="Bot", level=logging.DEBUG)
intercept_all_loggers()

ROOTLOGGER = logging.getLogger()
HANDLER_LOGGER = logging.getLogger("handler")

NAME_LOGGER = logging.getLogger(__name__)
logging.getLogger("asyncio").setLevel(logging.DEBUG)
logging.getLogger("aiomonitor").setLevel(logging.DEBUG)

INVITE_LINK = "https://discordapp.com/api/oauth2/authorize?client_id={}&scope=bot&permissions=0"


DESCRIPTION = """An example bot to showcase the discord.ext.commands extension
module.

There are a number of utility commands being showcased here."""

HOME_PATH = os.environ.get("HOME")

COMMAND_RUNNER = {"dl_thumb": shell.run_coroutine_subprocess}


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
async def get_prefix(_bot: "GoobBot", message: discord.message.Message):
    """_summary_

    Args:
        _bot (GoobBot): _description_
        message (discord.message.Message): _description_

    Returns:
        _type_: _description_
    """
    LOGGER.info(f"inside get_prefix(_bot, message) - > get_prefix({_bot}, {message})")
    LOGGER.info(f"inside get_prefix(_bot, message) - > get_prefix({_bot}, {message})")
    LOGGER.info(f"inside get_prefix(_bot, message) - > get_prefix({type(_bot)}, {type(message)})")
    prefix = (
        [aiosettings.prefix]
        if isinstance(message.channel, discord.DMChannel)
        else [utils.get_guild_prefix(_bot, message.guild.id)]
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


class GoobBot(commands.AutoShardedBot):
    """_summary_

    Args:
        commands (_type_): _description_
    """

    def __init__(
        self,
        *args,
        # command_prefix=get_prefix,
        intents: discord.flags.Intents = discord.Intents.default(),
        # TEMPCHANGE: # command_prefix=commands.when_mentioned_or(constants.PREFIX),
        command_prefix=commands.when_mentioned_or(aiosettings.prefix),
        description="Better than the last one",
        **kwargs,
    ):
        """_summary_

        Args:
            intents (discord.flags.Intents, optional): _description_. Defaults to discord.Intents.default().
            command_prefix (_type_, optional): _description_. Defaults to commands.when_mentioned_or(constants.PREFIX).
            description (str, optional): _description_. Defaults to "Better than the last one".
        """
        # super().__init__(
        #     *args, command_prefix=command_prefix, description=description, **kwargs
        # )
        # super(GoobBot, self).__init__(length, length)
        # https://realpython.com/python-super/#a-super-deep-dive
        super(GoobBot, self).__init__(
            *args,
            intents=intents,
            command_prefix=command_prefix,
            description=description,
            **kwargs,
        )

        # self.session = aiohttp.ClientSession(loop=self.loop)

        # Create a queue that we will use to store our "workload".
        self.queue = asyncio.Queue()

        self.tasks = []

        self.num_workers = 3

        self.total_sleep_time = 0

        self.start_time = datetime.datetime.now()

        # self.metrics_api = GoobBotMetricsApi(metrics_host="0.0.0.0")

        # DISABLED: 3/25/2023 temporary to figure some stuff out # self.loop.run_until_complete(self.metrics_api.start())

        self.typerCtx = None

        #### For upscaler

        self.job_queue = {}

        # This group of variables are used in the upscaling process
        self.last_model = None
        self.last_in_nc = None
        self.last_out_nc = None
        self.last_nf = None
        self.last_nb = None
        self.last_scale = None
        self.last_kind = None
        self.model = None
        self.autocrop_model = None
        self.db = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        self.ml_models_path = f"{aiosettings.esgran_dir}"
        self.my_custom_ml_models_path = f"{aiosettings.screencropnet_dir}/"

        self.current_task = None

    async def setup_hook(self) -> None:
        """_summary_"""

        self.version = goob_ai.__version__
        self.guild_data = {}
        self.intents.members = True
        self.intents.message_content = True

        self.db = db.init_worker_redis()

        # here, we are loading extensions prior to sync to ensure we are syncing interactions defined in those extensions.

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

    # # Method translated to python from BlueAmulet's original alias PR
    # # Basically this allows the fuzzy matching to look at individual phrases present in the model name
    # # This way, if you had two models, e.g 4xBox and 4x_sponge_bob, you could type 'bob' and it will choose the correct model
    # # This method just builds the alias dictionary and list for that functionality
    # def build_aliases(self):
    #     """Builds aliases for fuzzy string matching the model name input"""
    #     aliases = {}

    #     # Create aliases for models based on unique parts
    #     for model in self.models:
    #         name = os.path.splitext(os.path.basename(model))[0]
    #         parts = re.findall(r"([0-9]+x?|[A-Z]+(?![a-z])|[A-Z][^A-Z0-9_-]*)", name)
    #         for i in range(len(parts)):
    #             for j in range(i + 1, len(parts) + 1):
    #                 alias = "".join(parts[i:j])
    #                 if alias in aliases:
    #                     if fuzz.ratio(alias, model) > fuzz.ratio(alias, aliases[alias]):
    #                         aliases[alias] = model
    #                 else:
    #                     aliases[alias] = model

    #     # Ensure exact names are usable
    #     for model in self.models:
    #         name = os.path.splitext(os.path.basename(model))[0]
    #         aliases[name] = model

    #     fuzzylist = [alias for alias, value in aliases.items() if value]
    #     print(f"Made {len(fuzzylist)} aliases for {len(self.models)} models.")
    #     LOGGER.debug(f"Made {fuzzylist} aliases for {self.models} models.")
    #     return fuzzylist, aliases

    # SOURCE: https://discordpy.readthedocs.io/en/stable/api.html?highlight=event#discord.on_ready
    async def on_ready(self) -> None:
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

    async def my_background_task(self) -> None:
        """_summary_"""
        await self.wait_until_ready()
        counter = 0
        # TEMPCHANGE: # channel = self.get_channel(DISCORD_GENERAL_CHANNEL)  # channel ID goes here
        channel = self.get_channel(aiosettings.discord_general_channel)  # channel ID goes here
        while not self.is_closed():
            counter += 1
            await channel.send(counter)
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


# _bot: "GoobBot", message: discord.message.Message


# def load_extensions(_bot: "GoobBot") -> None:
#     """_summary_

#     Args:
#         _bot (GoobBot): _description_
#     """
#     for ext in extensions():
#         try:
#             _bot.load_extension(ext)
#         except Exception as ex:
#             print(f"Failed to load extension {ext} - exception: {ex}")
#             exc_type, exc_value, exc_traceback = sys.exc_info()
#             LOGGER.error(f"Error Class: {str(ex.__class__)}")
#             output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
#             LOGGER.warning(output)
#             LOGGER.error(f"exc_type: {exc_type}")
#             LOGGER.error(f"exc_value: {exc_value}")
#             traceback.print_tb(exc_traceback)
#             raise


if __name__ == "__main__":
    # GoobBot = GoobBot()
    # # goob_ai/bot.py:528:4: E0237: Assigning to attribute 'members' not defined in class slots (assigning-non-slot)
    # GoobBot.intents.members = True  # pylint: disable=assigning-non-slot
    # # NOTE: https://github.com/makupi/cookiecutter-discord.py-postgres/blob/master/%7B%7Bcookiecutter.bot_slug%7D%7D/bot/__init__.py
    # GoobBot.version = goob_ai.__version__
    # GoobBot.guild_data = {}
    # # load_extensions(GoobBot)
    # # GoobBot.add_command(echo)
    # # import bpdb; bpdb.set_trace()
    # _cog = GoobBot.get_cog("Utility")
    # utility_commands = _cog.get_commands()
    # print([c.name for c in utility_commands])
    # # print([c.name for c in commands])

    # # it is possible to pass a dictionary with local variables
    # # to the python console environment
    # host, port = "localhost", 50101
    # locals_ = {"port": port, "host": host}
    # # init monitor just before run_app

    # # intents = discord.Intents.default()
    # # intents.members = True
    # # DiscordBackend.client = discord.Client(intents=intents)

    # # import rich
    # # # cli.CLI()
    # # temp_runner = CliRunner()
    # # result: ClickResult
    # # result = temp_runner.invoke(cli.CLI, ["dump-context"])
    # # rich.print("Lets see if that populated the CTX cached value we need")
    # # rich.print(cli.CACHE_CTX)
    # # # rich.print(result)
    # # import bpdb
    # # bpdb.set_trace()

    # aiodebug_log_slow_callbacks.enable(0.05)

    # with aiomonitor.start_monitor(loop=GoobBot.loop, locals=locals_):
    #     # TEMPCHANGE: GoobBot.run(DISCORD_TOKEN)
    #     # TEMPCHANGE: replacing DISCORD_TOKEN with aiosettings.discord_token
    #     GoobBot.run(aiosettings.discord_token)

    intents = discord.Intents.default()
    intents.message_content = True

    async def aio_smoke_test() -> None:
        async with GoobBot(intents=intents) as GoobBot:
            await GoobBot.start(aiosettings.discord_token)

    # For most use cases, after defining what needs to run, we can just tell asyncio to run it:
    asyncio.run(aio_smoke_test())
