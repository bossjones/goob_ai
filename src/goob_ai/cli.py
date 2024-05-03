"""goob_ai.cli"""
from __future__ import annotations

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

# from redis.asyncio import ConnectionPool, Redis
import rich
from rich.pretty import pprint
import typer

import goob_ai
from goob_ai import settings_validator
from goob_ai.aio_settings import aiosettings, config_to_table, get_rich_console
from goob_ai.bot import GoobBot, aiomonitor  # , load_extensions

# TEMPCHANGE: # from goob_ai.bot import DISCORD_TOKEN, GoobBot, aiomonitor, load_extensions
from goob_ai.bot_logger import get_logger

# type: ignore
# noqa: E402

# from goob_ai.utils.async_ import aobject

LOGGER = get_logger(__name__, provider="CLI", level=logging.DEBUG)

CACHE_CTX = {}


# # SOURCE: https://github.com/long2ice/fettler/blob/271bbf68e3c08cb02f285a7ff9e3f2357ce1216e/fettler/cli.py
# def coro(f):
#     @wraps(f)
#     def wrapper(*args, **kwargs):
#         loop = asyncio.get_event_loop()
#         loop.run_until_complete(f(*args, **kwargs))

#     return wrapper


async def co_get_ctx(ctxs: List[typer.Context]) -> None:
    """_summary_

    Args:
        ctxs (List[typer.Context]): _description_
    """
    await asyncio.sleep(1)
    for c in ctxs:
        CACHE_CTX[c.name] = c
    typer.echo(f"CACHE_CTX -> {CACHE_CTX}")


# async def co_get_ctx_improved(ctx: typer.Context):
#     await asyncio.sleep(1)
#     return ctx
#     # yield ctx


# When you create an CLI = typer.Typer() it works as a group of commands.
CLI = typer.Typer(name="cerebroctl", callback=CACHE_CTX)


@CLI.command()
def hello(ctx: typer.Context, name: Optional[str] = typer.Option(None)) -> None:
    """
    Dummy command
    """
    typer.echo(f"Hello {name}")


@CLI.command()
def dump_context(ctx: typer.Context) -> typer.Context:
    """
    Dump Context
    """
    typer.echo("\nDumping context:\n")
    pprint(ctx.meta)
    return ctx


@CLI.command()
def config_dump(ctx: typer.Context) -> typer.Context:
    """
    Dump GoobBot config info to STD out
    """
    assert ctx.meta
    typer.echo("\nDumping config:\n")
    console = get_rich_console()
    config_to_table(console, aiosettings)


@CLI.command()
def doctor(ctx: typer.Context) -> typer.Context:
    """
    Doctor checks your environment to verify it is ready to run GoobBot
    """
    assert ctx.meta
    typer.echo("\nRunning Doctor ...\n")
    settings_validator.get_rich_pretty_env_info()


@CLI.command()
def async_dump_context(ctx: typer.Context) -> None:
    """
    Dump Context
    """
    typer.echo("\nDumping context:\n")
    pprint(ctx.meta)
    asyncio.run(co_get_ctx(ctx))


@CLI.command()
def run(ctx: typer.Context) -> None:
    """
    Run cerebro bot
    """

    # # SOURCE: http://click.palletsprojects.com/en/7.x/commands/?highlight=__main__
    # # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # # by means other than the `if` block below
    # # ctx.ensure_object(dict)

    # typer.echo("\nStarting bot...\n")
    # cerebro = GoobBot()
    # # goob_ai/bot.py:528:4: E0237: Assigning to attribute 'members' not defined in class slots (assigning-non-slot)
    # cerebro.intents.members = True  # pylint: disable=assigning-non-slot
    # # NOTE: https://github.com/makupi/cookiecutter-discord.py-postgres/blob/master/%7B%7Bcookiecutter.bot_slug%7D%7D/bot/__init__.py
    # cerebro.version = goob_ai.__version__
    # cerebro.guild_data = {}
    # cerebro.typerCtx = ctx
    # load_extensions(cerebro)
    # _cog = cerebro.get_cog("Utility")
    # utility_commands = _cog.get_commands()
    # print([c.name for c in utility_commands])

    # # TEMPCHANGE: 3/26/2023 - Trying to see if it loads settings in time.
    # # TEMPCHANGE: # it is possible to pass a dictionary with local variables
    # # TEMPCHANGE: # to the python console environment
    # # TEMPCHANGE: host, port = "localhost", 50101
    # # TEMPCHANGE: locals_ = {"port": port, "host": host}

    # locals_ = aiosettings.aiomonitor_config_data

    # # aiodebug_log_slow_callbacks.enable(0.05)
    # with aiomonitor.start_monitor(loop=cerebro.loop, locals=locals_):
    #     cerebro.run(aiosettings.discord_token)
    # run_async(aio_go_run_cerebro())
    intents = discord.Intents.default()
    intents.message_content = True

    async def run_cerebro() -> None:
        async with GoobBot(intents=intents) as cerebro:
            cerebro.typerCtx = ctx
            await cerebro.start(aiosettings.discord_token)

    # For most use cases, after defining what needs to run, we can just tell asyncio to run it:
    asyncio.run(run_cerebro())


async def cerebro_initialized(ctx: typer.Context) -> None:
    """_summary_

    Args:
        ctx (_type_): _description_
    """
    ctx.ensure_object(dict)
    await asyncio.sleep(1)
    print("GoobBot Context Ready to Go")


class GoobBotContext:
    """_summary_

    Returns:
        _type_: _description_
    """

    @classmethod
    async def create(
        cls,
        ctx: typer.Context,
        debug: bool = True,
        run_bot: bool = True,
        run_web: bool = True,
        run_metrics: bool = True,
        run_aiodebug: bool = False,
    ) -> "GoobBotContext":
        """_summary_

        Args:
            ctx (typer.Context): _description_
            debug (bool, optional): _description_. Defaults to True.
            run_bot (bool, optional): _description_. Defaults to True.
            run_web (bool, optional): _description_. Defaults to True.
            run_metrics (bool, optional): _description_. Defaults to True.
            run_aiodebug (bool, optional): _description_. Defaults to False.

        Returns:
            GoobBotContext: _description_
        """
        # ctx.ensure_object(dict)
        await cerebro_initialized(ctx)
        self = GoobBotContext()
        self.ctx = ctx
        self.debug = debug
        self.run_bot = run_bot
        self.run_web = run_web
        self.run_metrics = run_metrics
        self.run_aiodebug = run_aiodebug
        self.debug = debug
        return self


@CLI.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "auto_envvar_prefix": "CEREBRO",
    }
)
@CLI.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    debug: bool = True,
    run_bot: bool = True,
    run_web: bool = True,
    run_metrics: bool = True,
    run_aiodebug: bool = False,
) -> typer.Context:
    """
    Manage users in the awesome CLI app.
    """

    ctx.ensure_object(dict)
    # ctx.obj = await GoobBotContext.create(
    #     ctx, debug, run_bot, run_web, run_metrics, run_aiodebug
    # )

    # SOURCE: http://click.palletsprojects.com/en/7.x/commands/?highlight=__main__
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    # asyncio.run(co_get_ctx_improved(ctx))

    ctx.meta["DEBUG"] = debug
    ctx.obj["DEBUG"] = debug

    ctx.meta["AIOMONITOR"] = run_metrics
    ctx.obj["AIOMONITOR"] = run_metrics

    for extra_arg in ctx.args:
        typer.echo(f"Got extra arg: {extra_arg}")
    typer.echo(f"About to execute command: {ctx.invoked_subcommand}")
    return ctx


# SOURCE: servox
def run_async(future: Union[asyncio.Future, asyncio.Task, Awaitable]) -> Any:
    """Run the asyncio event loop until Future is done.

    This function is a convenience alias for `asyncio.get_event_loop().run_until_complete(future)`.

    Args:
        future: The future to run.

    Returns:
        Any: The Future's result.

    Raises:
        Exception: Any exception raised during execution of the future.
    """
    return asyncio.get_event_loop().run_until_complete(future)


if __name__ == "__main__":
    _ctx = CLI()
    rich.print("CTX")
    rich.print(_ctx)

#  autoflake --imports=dropbox,discord,unicodedata,six,uritools,goob_ai --in-place --remove-unused-variables goob_ai/cli.py
#  autoflake --imports=dropbox,discord,unicodedata,six,uritools,goob_ai --in-place --remove-unused-variables goob_ai/cli.py
#  autoflake --imports=dropbox,discord,unicodedata,six,uritools,goob_ai --in-place --remove-unused-variables goob_ai/cli.py
#  autoflake --imports=dropbox,discord,unicodedata,six,uritools,goob_ai --in-place --remove-unused-variables goob_ai/cli.py
#  autoflake --imports=dropbox,discord,unicodedata,six,uritools,goob_ai --in-place --remove-unused-variables goob_ai/cli.py
#  autoflake --imports=dropbox,discord,unicodedata,six,uritools,goob_ai --in-place --remove-unused-variables goob_ai/cli.py
