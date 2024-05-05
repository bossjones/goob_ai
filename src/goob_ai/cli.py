"""goob_ai.cli"""

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
from goob_ai.aio_settings import aiosettings, config_to_table, get_rich_console
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from goob_ai.bot_logger import get_logger

from typing import Any, Dict, Optional, Tuple

import rich
import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated


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


LOGGER = get_logger(__name__, provider="CLI", level=logging.DEBUG)


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


if __name__ == "__main__":
    APP()
