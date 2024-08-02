"""goob_ai.utils.asynctyper"""

from __future__ import annotations

import asyncio
import inspect
import logging

from collections.abc import Awaitable, Callable, Iterable, Sequence
from functools import partial, wraps
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import anyio
import asyncer
import discord
import rich
import typer

from rich.pretty import pprint
from typer import Typer

import goob_ai

from goob_ai.aio_settings import aiosettings, get_rich_console
from goob_ai.bot_logger import get_logger


F = TypeVar("F", bound=Callable[..., Any])


class AsyncTyper(Typer):
    """
    A custom Typer class that supports asynchronous functions.

    This class decorates functions with the given decorator, but only if the function
    is not already a coroutine function.
    """

    @staticmethod
    def maybe_run_async(decorator: Callable[[F], F], f: F) -> F:
        """
        Decorates a function with the given decorator if it's not a coroutine function.

        Args:
            decorator: The decorator to apply to the function.
            f: The function to decorate.

        Returns:
            The decorated function.
        """
        if inspect.iscoroutinefunction(f):

            @wraps(f)
            def runner(*args: Any, **kwargs: Any) -> Any:
                return asyncer.runnify(f)(*args, **kwargs)

            decorator(runner)
        else:
            decorator(f)
        return f

    def callback(self, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        """
        Overrides the callback method to support asynchronous functions.

        Returns:
            A partial function that applies the async decorator to the callback.
        """
        decorator = super().callback(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)

    def command(self, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        """
        Overrides the command method to support asynchronous functions.

        Returns:
            A partial function that applies the async decorator to the command.
        """
        decorator = super().command(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)
