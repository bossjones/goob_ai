"""goob_ai.cli"""

from __future__ import annotations

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
from goob_ai.aio_settings import aiosettings, config_to_table, get_rich_console
from goob_ai.bot import GoobBot, aiomonitor  # , load_extensions
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from goob_ai.bot_logger import get_logger

from typing import Any, Dict, Optional, Tuple

import rich
import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

# SOURCE: https://github.com/Ce11an/tfl/blob/76b07a6f465e9c41c11f9eb812dc7d82e226ab3b/tfl/cli/main.py

#####################
# Trying to use AsyncTyper instead of old cerebocode

# noqa: E402


# LOGGER = get_logger(__name__, provider="CLI", level=logging.DEBUG)

# CACHE_CTX: Optional[Callable[..., Any]] = {}


# async def co_get_ctx(ctxs: List[typer.Context]) -> None:
#     """_summary_

#     Args:
#         ctxs (List[typer.Context]): _description_
#     """
#     await asyncio.sleep(1)
#     for c in ctxs:
#         CACHE_CTX[c.name] = c
#     typer.echo(f"CACHE_CTX -> {CACHE_CTX}")


# # When you create an CLI = typer.Typer() it works as a group of commands.
# CLI = typer.Typer(name="cerebroctl", callback=CACHE_CTX)


# @CLI.command()
# def hello(ctx: typer.Context, name: Optional[str] = typer.Option(None)) -> None:
#     """
#     Dummy command
#     """
#     typer.echo(f"Hello {name}")


# @CLI.command()
# def dump_context(ctx: typer.Context) -> typer.Context:
#     """
#     Dump Context
#     """
#     typer.echo("\nDumping context:\n")
#     pprint(ctx.meta)
#     return ctx


# @CLI.command()
# def config_dump(ctx: typer.Context) -> typer.Context:
#     """
#     Dump GoobBot config info to STD out
#     """
#     assert ctx.meta
#     typer.echo("\nDumping config:\n")
#     console = get_rich_console()
#     config_to_table(console, aiosettings)


# @CLI.command()
# def doctor(ctx: typer.Context) -> typer.Context:
#     """
#     Doctor checks your environment to verify it is ready to run GoobBot
#     """
#     assert ctx.meta
#     typer.echo("\nRunning Doctor ...\n")
#     settings_validator.get_rich_pretty_env_info()


# @CLI.command()
# def async_dump_context(ctx: typer.Context) -> None:
#     """
#     Dump Context
#     """
#     typer.echo("\nDumping context:\n")
#     pprint(ctx.meta)
#     asyncio.run(co_get_ctx(ctx))


# @CLI.command()
# def run(ctx: typer.Context) -> None:
#     """
#     Run cerebro bot
#     """

#     intents = discord.Intents.default()
#     intents.message_content = True

#     async def run_cerebro() -> None:
#         async with GoobBot(intents=intents) as cerebro:
#             cerebro.typerCtx = ctx
#             await cerebro.start(aiosettings.discord_token)

#     # For most use cases, after defining what needs to run, we can just tell asyncio to run it:
#     asyncio.run(run_cerebro())


# async def cerebro_initialized(ctx: typer.Context) -> None:
#     """_summary_

#     Args:
#         ctx (_type_): _description_
#     """
#     ctx.ensure_object(dict)
#     await asyncio.sleep(1)
#     print("GoobBot Context Ready to Go")


# class GoobBotContext:
#     """_summary_

#     Returns:
#         _type_: _description_
#     """

#     @classmethod
#     async def create(
#         cls,
#         ctx: typer.Context,
#         debug: bool = True,
#         run_bot: bool = True,
#         run_web: bool = True,
#         run_metrics: bool = True,
#         run_aiodebug: bool = False,
#     ) -> "GoobBotContext":
#         """_summary_

#         Args:
#             ctx (typer.Context): _description_
#             debug (bool, optional): _description_. Defaults to True.
#             run_bot (bool, optional): _description_. Defaults to True.
#             run_web (bool, optional): _description_. Defaults to True.
#             run_metrics (bool, optional): _description_. Defaults to True.
#             run_aiodebug (bool, optional): _description_. Defaults to False.

#         Returns:
#             GoobBotContext: _description_
#         """
#         # ctx.ensure_object(dict)
#         await cerebro_initialized(ctx)
#         self = GoobBotContext()
#         self.ctx = ctx
#         self.debug = debug
#         self.run_bot = run_bot
#         self.run_web = run_web
#         self.run_metrics = run_metrics
#         self.run_aiodebug = run_aiodebug
#         self.debug = debug
#         return self


# @CLI.command(
#     context_settings={
#         "allow_extra_args": True,
#         "ignore_unknown_options": True,
#         "auto_envvar_prefix": "CEREBRO",
#     }
# )
# @CLI.callback(invoke_without_command=True)
# def main(
#     ctx: typer.Context,
#     debug: bool = True,
#     run_bot: bool = True,
#     run_web: bool = True,
#     run_metrics: bool = True,
#     run_aiodebug: bool = False,
# ) -> typer.Context:
#     """
#     Manage users in the awesome CLI app.
#     """

#     ctx.ensure_object(dict)
#     # ctx.obj = await GoobBotContext.create(
#     #     ctx, debug, run_bot, run_web, run_metrics, run_aiodebug
#     # )

#     # SOURCE: http://click.palletsprojects.com/en/7.x/commands/?highlight=__main__
#     # ensure that ctx.obj exists and is a dict (in case `cli()` is called
#     # by means other than the `if` block below
#     # asyncio.run(co_get_ctx_improved(ctx))

#     ctx.meta["DEBUG"] = debug
#     ctx.obj["DEBUG"] = debug

#     ctx.meta["AIOMONITOR"] = run_metrics
#     ctx.obj["AIOMONITOR"] = run_metrics

#     for extra_arg in ctx.args:
#         typer.echo(f"Got extra arg: {extra_arg}")
#     typer.echo(f"About to execute command: {ctx.invoked_subcommand}")
#     return ctx


# # SOURCE: servox
# def run_async(future: Union[asyncio.Future, asyncio.Task, Awaitable]) -> Any:
#     """Run the asyncio event loop until Future is done.

#     This function is a convenience alias for `asyncio.get_event_loop().run_until_complete(future)`.

#     Args:
#         future: The future to run.

#     Returns:
#         Any: The Future's result.

#     Raises:
#         Exception: Any exception raised during execution of the future.
#     """
#     return asyncio.get_event_loop().run_until_complete(future)


# if __name__ == "__main__":
#     _ctx = CLI()
#     rich.print("CTX")
#     rich.print(_ctx)


#########################################################

# SOURCE: https://github.com/tiangolo/typer/issues/88#issuecomment-1732469681
import json
import logging
import sys

import typer
from rich import print, print_json

from goob_ai.utils.asynctyper import AsyncTyper

# from ..bot import bot
# from ..config import settings

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)


APP = AsyncTyper()


@APP.command()
def about() -> None:
    typer.echo("This is a bot created from aulasoftwarelibre/telegram-bot-template")


# @app.async_command()
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


# @app.async_command()
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


@app.async_command()
async def serve() -> None:
    """Run polling bot version."""
    logging.info("Starting...")

    await bot.remove_webhook()
    await bot.infinity_polling(logger_level=logging.INFO)

    await bot.close_session()


# @app.async_command()
# async def uninstall() -> None:
#     """Uninstall bot webhook."""
#     await bot.remove_webhook()

#     await bot.close_session()


if __name__ == "__main__":
    app()
