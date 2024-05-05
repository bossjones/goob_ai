# history

```

pdm add boto3==1.34.41
pdm add fastapi==0.110.0


pdm add -dG test black==24.3.0 && \
pdm add -dG test boto3-stubs\[essential\]==1.34.79 && \
pdm add -dG test bpython==0.24 && \
pdm add -dG test flake8 && \
pdm add -dG test isort==5.13.2 && \

pdm add -dG test mypy==1.8.0 && \
pdm add -dG test mypy-boto3==1.34.79 && \
pdm add -dG test pre-commit==3.2.2 && \
pdm add -dG test pydocstyle==6.3.0 && \
pdm add -dG test pytest-cov==4.1.0 && \
pdm add -dG test pytest-mock==3.12.0 && \
pdm add -dG test pytest-sugar==1.0.0 && \
pdm add -dG test pytest==8.0.0 && \
pdm add -dG test pyupgrade==3.15.2 && \
pdm add -dG test requests_mock==1.11.0 && \
pdm add -dG test rich==13.7.1 && \
pdm add -dG test ruff==0.3.7 && \
pdm add -dG test types-beautifulsoup4==4.12.0.20240229 && \
pdm add -dG test types-boto==2.49.18.20240205 && \
pdm add -dG test types-mock==5.1.0.20240311 && \
pdm add -dG test "types-requests<2.31.0.7"






pdm add -dG test validate-pyproject\[all,store\]==0.16 && \
pdm add beautifulsoup4==4.10.0 && \
pdm add chardet==5.2.0 && \
pdm add langchain_community==0.0.33 && \
pdm add langchain_openai==0.0.8 && \
pdm add langchain==0.1.16 && \
pdm add openai==1.10.0 && \
pdm add pydantic-settings==2.1.0 && \
pdm add pydantic==2.6.1 && \
pdm add pypdf==4.0.1 && \
pdm add python-docx==1.1.0 && \
pdm add python-dotenv==1.0.1 && \
pdm add striprtf==0.0.26 && \
pdm add tenacity==8.1.0 && \
pdm add requests==v2.31.0 && \
pdm add uvicorn==0.28.0


# FROM cerebro, base

pdm add aiocache && \
pdm add aiodebug && \
pdm add aiodns && \
pdm add aiofile && \
pdm add aiofiles && \
pdm add aiohttp && \
pdm add aiohttp-json-rpc && \
pdm add aiomonitor && \
pdm add aioprometheus\[starlette\] && \
pdm add aioreactive && \
pdm add aiosql && \
pdm add aiosqlite && \
pdm add attrs && \
pdm add better_exceptions && \
pdm add click-spinner && \
pdm add codetiming && \
pdm add discord.py && \
pdm add dropbox && \
pdm add factory-boy && \
pdm add faker && \
pdm add fonttools\[woff\] && \
pdm add fuzzywuzzy\[speedup\] && \
pdm add gallery-dl && \
pdm add google-auth && \
pdm add google-auth-oauthlib && \
pdm add gutter && \
pdm add html5lib && \
pdm add imageio && \
pdm add imutils && \
pdm add logging_tree && \
pdm add loguru && \
pdm add lxml && \
pdm add markdown && \
pdm add markdownify && \
pdm add matplotlib && \
pdm add md2pdf && \
pdm add memory_profiler && \
pdm add motor && \
pdm add multiprocess && \
pdm add mutagen && \
pdm add numpy && \
pdm add passlib\[bcrypt\] && \
pdm add Pillow && \
pdm add prettytable && \
pdm add pycryptodome && \
pdm add pygments && \
pdm add pyinspect && \
pdm add PyPDF2 && \
pdm add pypi-command-line\[speedups\\] && \
pdm add pytablewriter[html\] && \
pdm add python-Levenshtein && \
pdm add python-slugify && \
pdm add pytz && \
pdm add redis && \
pdm add scenedetect\[opencv\] && \
pdm add sentencepiece && \
pdm add simpletransformers && \
pdm add soupsieve && \
pdm add streamlit && \
pdm add telnetlib3 && \
pdm add tenacity && \
pdm add tomli && \
pdm add tqdm && \
pdm add transformers && \
pdm add typer && \
pdm add Unidecode && \
pdm add uritemplate && \
pdm add uritools && \
pdm add validators && \
pdm add watchdog && \
pdm add webcolors && \
pdm add websockets && \
pdm add youtube_dl && \
pdm add yt-dlp



pdm add docutils  && \
pdm add pyinvoke  && \
pdm add pathlib_mate  && \
pdm add lazy-object-proxy  && \
pdm add -dG test types-PyYAML && \
pdm add -dG test types-aiofiles && \
pdm add -dG test types-click && \
pdm add -dG test types-colorama && \
pdm add -dG test types-dataclasses && \
pdm add -dG test types-freezegun && \
pdm add -dG test types-mock && \
pdm add -dG test types-pytz && \
pdm add -dG test types-requests && \
pdm add -dG test types-setuptools && \
pdm add -dG test types-six && \
pdm add -dG test MonkeyType && \
pdm add -dG test hunter && \
pdm add -dG test sourcery




pdm add httpx\[http2\]  && \
pdm add -dG test types-beautifulsoup4==4.12.0.20240229 && \
pdm add -dG test types-html5lib==1.1.11.20240228 && \
pdm add -dG test types-pillow==10.2.0.20240213 && \
pdm add -dG test types-pyasn1==0.6.0.20240402 && \
pdm add -dG test types-python-jose==3.3.4.20240106 && \
pdm add -dG test types-pytz==2024.1.0.20240203 && \
pdm add -dG test typing-extensions==4.10.0
pdm add -dG test pylint-per-file-ignores


pdm add -dG test pyright



pdm add -dG test pytest-rerunfailures
pdm add -dG test pytest-asyncio

pdm add python-json-logger

pdm add transformers && \
pdm add chromadb && \
pdm add duckduckgo-search && \
pdm add wikipedia && \
pdm add youtube-transcript-api && \
pdm add torch

pdm add pydantic\[dotenv,email\]
pdm add pyinvoke


pdm install -Gtest -d
```


# use venv

```bash
eval $(pdm venv activate)
```


## pyright

https://github.com/microsoft/pyright/blob/main/docs/configuration.md


## Old cli command

```

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
#     Manage users in the awesome CLI APP.
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


# from ..bot import bot
# from ..config import settings

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
```
