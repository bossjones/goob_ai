"""goob_ai.cli"""

# pylint: disable=no-value-for-parameter
# SOURCE: https://github.com/tiangolo/typer/issues/88#issuecomment-1732469681
from __future__ import annotations

import json
import sys
from importlib import import_module, metadata
import subprocess
import os

from pathlib import Path
import inspect
from functools import partial, wraps

import anyio
import asyncer
import typer
from typer import Typer
import asyncio
import logging
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    Union,
)

import discord

import rich
from rich.pretty import pprint
import typer

import goob_ai
from goob_ai import settings_validator
from goob_ai.aio_settings import aiosettings, get_rich_console
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from goob_ai.goob_bot import AsyncGoobBot

from goob_ai.bot_logger import get_logger

from typing import Any, Dict, Optional, Tuple

import rich
import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from loguru import logger

import json
import logging
import sys

import typer
from rich import print, print_json

from goob_ai.asynctyper import AsyncTyper
from typing import Any, Dict, Optional, Tuple

import rich
import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated


# LOGGER = get_logger(__name__, provider="CLI", level=logging.DEBUG)

from goob_ai.bot_logger import get_logger, global_log_config

global_log_config(
    log_level=logging.getLevelName("DEBUG"),
    json=False,
)

LOGGER = logger


APP = AsyncTyper()
console = Console()


# Load existing subcommands
def load_commands(directory: str = "subcommands"):
    script_dir = Path(__file__).parent
    subcommands_dir = script_dir / directory

    LOGGER.info(f"Loading subcommands from {subcommands_dir}")

    for filename in os.listdir(subcommands_dir):
        if filename.endswith("_cmd.py"):
            module_name = f'{__name__.split(".")[0]}.{directory}.{filename[:-3]}'
            module = import_module(module_name)
            if hasattr(module, "app"):
                APP.add_typer(module.app, name=filename[:-7])


def version_callback(version: bool) -> None:
    """Print the version of goob_ai."""
    if version:
        rich.print(f"goob_ai version: {goob_ai.__version__}")
        raise typer.Exit()


@APP.command()
def about() -> None:
    """about command"""
    typer.echo("This is GoobBot CLI")


# @APP.async_command()
# async def info() -> None:
#     """Returns information about the bot."""
#     result = await bot.get_me()
#     print("Bot me information")
#     print_json(result.to_json())
#     result = await bot.get_webhook_info()
#     print("Bot webhook information")
#     print_json(
#         json.dumps(
#             {
#                 "url": result.url,
#                 "has_custom_certificate": result.has_custom_certificate,
#                 "pending_update_count": result.pending_update_count,
#                 "ip_address": result.ip_address,
#                 "last_error_date": result.last_error_date,
#                 "last_error_message": result.last_error_message,
#                 "last_synchronization_error_date": result.last_synchronization_error_date,
#                 "max_connections": result.max_connections,
#                 "allowed_updates": result.allowed_updates,
#             }
#         )
#     )
#     await bot.close_session()


# @APP.async_command()
# async def install() -> None:
#     """Install bot webhook"""
#     # Remove webhook, it fails sometimes the set if there is a previous webhook
#     await bot.remove_webhook()

#     WEBHOOK_URL_BASE = f"https://{settings.webhook_host}:{443}"
#     WEBHOOK_URL_PATH = f"/{settings.secret_token}/"

#     # Set webhook
#     result = await bot.set_webhook(url=WEBHOOK_URL_BASE + WEBHOOK_URL_PATH)

#     print(f"Set webhook to {WEBHOOK_URL_BASE + WEBHOOK_URL_PATH}: {result}")

#     await bot.close_session()


# @APP.async_command()
# async def serve() -> None:
#     """Run polling bot version."""
#     logging.info("Starting...")

#     await bot.remove_webhook()
#     await bot.infinity_polling(logger_level=logging.INFO)

#     await bot.close_session()


# # @APP.async_command()
# # async def uninstall() -> None:
# #     """Uninstall bot webhook."""
# #     await bot.remove_webhook()


# #     await bot.close_session()
def main():
    APP()
    load_commands()


def entry():
    """Required entry point to enable hydra to work as a console_script."""
    main()  # pylint: disable=no-value-for-parameter


async def run_bot():
    async with AsyncGoobBot() as bot:
        # bot.typerCtx = ctx
        # bot.typerCtx = ctx
        # bot.pool = pool
        await bot.start(aiosettings.discord_token)
    # log = logging.getLogger()
    # try:
    #     pool = await create_pool()
    # except Exception:
    #     click.echo('Could not set up PostgreSQL. Exiting.', file=sys.stderr)
    #     log.exception('Could not set up PostgreSQL. Exiting.')
    #     return

    # async with RoboDanny() as bot:
    #     bot.pool = pool
    #     await bot.start()


# @click.group(invoke_without_command=True, options_metavar='[options]')
# @click.pass_context
# def main(ctx):
#     """Launches the bot."""
#     if ctx.invoked_subcommand is None:
#         with setup_logging():
#             asyncio.run(run_bot())


@APP.command()
def go() -> None:
    """Main entry point for goobbot"""
    typer.echo("Starting up GoobAI Bot")
    asyncio.run(run_bot())


if __name__ == "__main__":
    APP()


# TODO: Add this
# @CLI.command()
# def run(ctx: typer.Context) -> None:
#     """
#     Run cerebro bot
#     """

#     # # SOURCE: http://click.palletsprojects.com/en/7.x/commands/?highlight=__main__
#     # # ensure that ctx.obj exists and is a dict (in case `cli()` is called
#     # # by means other than the `if` block below
#     # # ctx.ensure_object(dict)

#     # typer.echo("\nStarting bot...\n")
#     # cerebro = Cerebro()
#     # # cerebro_bot/bot.py:528:4: E0237: Assigning to attribute 'members' not defined in class slots (assigning-non-slot)
#     # cerebro.intents.members = True  # pylint: disable=assigning-non-slot
#     # # NOTE: https://github.com/makupi/cookiecutter-discord.py-postgres/blob/master/%7B%7Bcookiecutter.bot_slug%7D%7D/bot/__init__.py
#     # cerebro.version = cerebro_bot.__version__
#     # cerebro.guild_data = {}
#     # cerebro.typerCtx = ctx
#     # load_extensions(cerebro)
#     # _cog = cerebro.get_cog("Utility")
#     # utility_commands = _cog.get_commands()
#     # print([c.name for c in utility_commands])

#     # # TEMPCHANGE: 3/26/2023 - Trying to see if it loads settings in time.
#     # # TEMPCHANGE: # it is possible to pass a dictionary with local variables
#     # # TEMPCHANGE: # to the python console environment
#     # # TEMPCHANGE: host, port = "localhost", 50101
#     # # TEMPCHANGE: locals_ = {"port": port, "host": host}

#     # locals_ = aiosettings.aiomonitor_config_data

#     # # aiodebug_log_slow_callbacks.enable(0.05)
#     # with aiomonitor.start_monitor(loop=cerebro.loop, locals=locals_):
#     #     cerebro.run(aiosettings.discord_token)
#     # run_async(aio_go_run_cerebro())
#     intents = discord.Intents.default()
#     intents.message_content = True

#     async def run_cerebro() -> None:
#         async with Cerebro(intents=intents) as cerebro:
#             cerebro.typerCtx = ctx
#             await cerebro.start(aiosettings.discord_token)

#     # For most use cases, after defining what needs to run, we can just tell asyncio to run it:
#     asyncio.run(run_cerebro())
