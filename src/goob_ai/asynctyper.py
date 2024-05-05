"""goob_ai.utils.asynctyper"""

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


class AsyncTyper(Typer):
    """
    Decorates a function with the given decorator, but only if the function is not already a coroutine function.

    Args:
        decorator (function): The decorator to apply to the function.
        f (function): The function to decorate.

    Returns:
        function: The decorated function.
    """

    @staticmethod
    def maybe_run_async(decorator, f):
        if inspect.iscoroutinefunction(f):

            @wraps(f)
            def runner(*args, **kwargs):
                return asyncer.runnify(f)(*args, **kwargs)

            decorator(runner)
        else:
            decorator(f)
        return f

    def callback(self, *args, **kwargs):
        decorator = super().callback(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)

    def command(self, *args, **kwargs):
        decorator = super().command(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)


# app = AsyncTyper()


# @app.command()
# async def async_hello(name: str, last_name: str = "") -> None:
#     await anyio.sleep(1)
#     typer.echo(f"Hello World {name} {last_name}")


# @app.command()
# def hello() -> None:
#     print("Hello World")


# if __name__ == "__main__":
#     app()
