"""goob_ai.cli"""

# pyright: reportMissingTypeStubs=false
# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# SOURCE: https://github.com/tiangolo/typer/issues/88#issuecomment-1732469681
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
import traceback
import typing

from collections.abc import Awaitable, Iterable, Sequence
from enum import Enum
from functools import partial, wraps
from importlib import import_module, metadata
from importlib.metadata import version as importlib_metadata_version
from pathlib import Path
from re import Pattern
from typing import Annotated, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import anyio
import asyncer
import bpdb
import discord
import rich
import sentry_sdk
import typer

from loguru import logger as LOGGER
from pinecone import Pinecone, ServerlessSpec  # pyright: ignore[reportAttributeAccessIssue]
from pinecone.core.openapi.data.model.describe_index_stats_response import DescribeIndexStatsResponse
from pinecone.core.openapi.data.model.query_response import QueryResponse
from pinecone.core.openapi.data.model.upsert_response import UpsertResponse
from pinecone.data.index import Index
from redis.asyncio import ConnectionPool, Redis
from rich import print, print_json
from rich.console import Console
from rich.pretty import pprint
from rich.prompt import Prompt
from rich.table import Table
from sentry_sdk.integrations.argv import ArgvIntegration
from sentry_sdk.integrations.atexit import AtexitIntegration
from sentry_sdk.integrations.dedupe import DedupeIntegration
from sentry_sdk.integrations.excepthook import ExcepthookIntegration
from sentry_sdk.integrations.modules import ModulesIntegration
from sentry_sdk.integrations.stdlib import StdlibIntegration
from sentry_sdk.integrations.threading import ThreadingIntegration
from typer import Typer

import goob_ai

from goob_ai import db
from goob_ai.aio_settings import aiosettings, get_rich_console
from goob_ai.asynctyper import AsyncTyper
from goob_ai.bot_logger import get_logger, global_log_config
from goob_ai.goob_bot import AsyncGoobBot
from goob_ai.monitoring.sentry import sentry_init
from goob_ai.services.chroma_service import ChromaService
from goob_ai.services.screencrop_service import ImageService
from goob_ai.utils import repo_typing
from goob_ai.utils.file_functions import fix_path


# # Use the following to enable the debugger
# from IPython.terminal.debugger import TerminalPdb  # noqa
# sys.excepthook = TerminalPdb(
#     call_pdb=True, ostream=sys.__stdout__
# )
# # sys.excepthook = TerminalPdb(
# #     color_scheme="Linux", call_pdb=True, ostream=sys.__stdout__
# # )

# if dev mode is enabled, set bpdb as the default debugger
if aiosettings.dev_mode:

    def info(type, value, tb):
        LOGGER.info(f"type: {type}")
        LOGGER.info(f"reveal_type(type): {typing.reveal_type(type)}")  # pylint: disable=undefined-variable

        LOGGER.info(f"value: {value}")
        LOGGER.info(f"reveal_type(value): {typing.reveal_type(value)}")  # pylint: disable=undefined-variable

        LOGGER.info(f"tb: {tb}")
        LOGGER.info(f"reveal_type(type): {typing.reveal_type(tb)}")  # pylint: disable=undefined-variable

        if hasattr(sys, "ps1") or not sys.stderr.isatty() or not sys.stdin.isatty():
            # stdin or stderr is redirected, just do the normal thing
            original_hook(type, value, tb)
        else:
            # a terminal is attached and stderr is not redirected, debug
            import traceback

            traceback.print_exception(type, value, tb)
            # environment variable PYTHON_DEBUG can select debugging type
            debug_style = os.environ.get("PYTHON_DEBUG", "bpdb")
            print(debug_style)
            if debug_style in ["pdb", "bpdb"]:
                print("[NOTE] automatic debugging from %s" % __file__, file=sys.stderr)
            if debug_style == "bpdb":
                import bpdb

                bpdb.pm()
            elif debug_style == "pdb":
                import pdb

                pdb.pm()
            else:
                raise Exception('cannot interpret environment variable PYTHON_DEBUG: "%s"' % debug_style)

    # automatically debug unless stdout/stderr redirected via stack overflow
    # ! note that python3 has more rigid scopes so you might not see everything you want
    original_hook = sys.excepthook
    # setting PYTHON_DEBUG to NO suppresses any debugging
    if sys.excepthook == sys.__excepthook__ and os.environ.get("PYTHON_DEBUG", "pdb") not in ["NO", "no"]:
        # if someone already patched excepthook, let them win
        sys.excepthook = info


if aiosettings.enable_sentry:
    # NOTE: DISABLED: Currently till I update sentry_init and start using it more.
    # sentry_init(
    #     # Set traces_sample_rate to 1.0 to capture 100%
    #     # of transactions for performance monitoring.
    #     traces_sample_rate=1.0,
    #     # Set profiles_sample_rate to 1.0 to profile 100%
    #     # of sampled transactions.
    #     # We recommend adjusting this value in production.
    #     profiles_sample_rate=1.0,
    #     # Turn off the default logging integration, but keep the rest.
    #     default_integrations=False,
    #     integrations=[
    #         AtexitIntegration(),
    #         ArgvIntegration(),
    #         DedupeIntegration(),
    #         ExcepthookIntegration(),
    #         StdlibIntegration(),
    #         ModulesIntegration(),
    #         ThreadingIntegration(),
    #     ],
    # )
    sentry_init(
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
    )
    logging.getLogger("sentry_sdk").setLevel(logging.WARNING)


global_log_config(
    log_level=logging.getLevelName("DEBUG"),
    json=False,
)


class ChromaChoices(str, Enum):
    load = "load"
    generate = "generate"
    get_response = "get_response"


APP = AsyncTyper()
console = Console()
cprint = console.print


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
def version() -> None:
    """Version command"""
    rich.print(f"goob_ai version: {goob_ai.__version__}")


@APP.command()
def deps() -> None:
    """Deps command"""
    rich.print(f"goob_ai version: {goob_ai.__version__}")
    rich.print(f"langchain_version: {importlib_metadata_version('langchain')}")
    rich.print(f"langchain_community_version: {importlib_metadata_version('langchain_community')}")
    rich.print(f"langchain_core_version: {importlib_metadata_version('langchain_core')}")
    rich.print(f"langchain_openai_version: {importlib_metadata_version('langchain_openai')}")
    rich.print(f"langchain_text_splitters_version: {importlib_metadata_version('langchain_text_splitters')}")
    rich.print(f"langchain_chroma_version: {importlib_metadata_version('langchain_chroma')}")
    rich.print(f"chromadb_version: {importlib_metadata_version('chromadb')}")
    rich.print(f"langsmith_version: {importlib_metadata_version('langsmith')}")
    rich.print(f"pydantic_version: {importlib_metadata_version('pydantic')}")
    rich.print(f"pydantic_settings_version: {importlib_metadata_version('pydantic_settings')}")
    rich.print(f"ruff_version: {importlib_metadata_version('ruff')}")


@APP.command()
def about() -> None:
    """About command"""
    typer.echo("This is GoobBot CLI")


@APP.command()
def show() -> None:
    """Show command"""
    cprint("\nShow goob_ai", style="yellow")


def main():
    APP()
    load_commands()


def entry():
    """Required entry point to enable hydra to work as a console_script."""
    main()  # pylint: disable=no-value-for-parameter


async def run_bot():
    try:
        pool: ConnectionPool = db.get_redis_conn_pool()
    except Exception as ex:
        print(f"{ex}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"Error Class: {ex.__class__}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        print(output)
        print(f"exc_type: {exc_type}")
        print(f"exc_value: {exc_value}")
        traceback.print_tb(exc_traceback)
        if aiosettings.dev_mode:
            bpdb.pm()
    async with AsyncGoobBot() as bot:
        if aiosettings.enable_redis:
            bot.pool = pool
        await bot.start()

    await LOGGER.complete()


# SOURCE: https://docs.pinecone.io/guides/getting-started/quickstart
@APP.command()
def create_index_quickstart() -> None:
    """Create a pinecone index"""
    typer.echo("Creating pinecone index...")

    typer.echo("1. Initialize your client connection")
    pc = Pinecone(api_key=aiosettings.pinecone_api_key.get_secret_value())

    # 4. Create a serverless index
    typer.echo("2. Create a serverless index")
    pc.create_index(
        name=aiosettings.pinecone_index,
        dimension=8,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # 5. Upsert vectors
    typer.echo("3. Upsert vectors")
    index: Index = pc.Index(aiosettings.pinecone_index)

    ns1_upsert_resp: UpsertResponse = index.upsert(
        vectors=[
            {"id": "vec1", "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
            {"id": "vec2", "values": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]},
            {"id": "vec3", "values": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]},
            {"id": "vec4", "values": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]},
        ],
        namespace="ns1",
    )

    ns2_upsert_resp: UpsertResponse = index.upsert(
        vectors=[
            {"id": "vec5", "values": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]},
            {"id": "vec6", "values": [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]},
            {"id": "vec7", "values": [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]},
            {"id": "vec8", "values": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]},
        ],
        namespace="ns2",
    )

    # 6. Check the index
    typer.echo("4. Check the index")
    index_rsp: DescribeIndexStatsResponse = index.describe_index_stats()
    # Returns:
    # {'dimension': 8,
    #  'index_fullness': 0.0,
    #  'namespaces': {'ns1': {'vector_count': 4}, 'ns2': {'vector_count': 4}},
    #  'total_vector_count': 8}
    typer.echo("5. Run a similarity search")
    n1_results: QueryResponse = index.query(
        namespace="ns1", vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], top_k=3, include_values=True
    )

    n2_results: QueryResponse = index.query(
        namespace="ns2", vector=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], top_k=3, include_values=True
    )

    # Returns:
    # {'matches': [{'id': 'vec3',
    #               'score': 0.0,
    #               'values': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]},
    #              {'id': 'vec4',
    #               'score': 0.0799999237,
    #               'values': [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]},
    #              {'id': 'vec2',
    #               'score': 0.0800000429,
    #               'values': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]}],
    #  'namespace': 'ns1',
    #  'usage': {'read_units': 6}}
    # {'matches': [{'id': 'vec7',
    #               'score': 0.0,
    #               'values': [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]},
    #              {'id': 'vec8',
    #               'score': 0.0799999237,
    #               'values': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]},
    #              {'id': 'vec6',
    #               'score': 0.0799999237,
    #               'values': [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]}],
    #  'namespace': 'ns2',
    #  'usage': {'read_units': 6}}


# SOURCE: https://docs.pinecone.io/guides/getting-started/quickstart
@APP.command()
def delete_index_quickstart() -> None:
    """Delete a pinecone index"""
    typer.echo("Deleting pinecone index...")
    pc = Pinecone(api_key=aiosettings.pinecone_api_key.get_secret_value())
    pc.delete_index(aiosettings.pinecone_index)
    typer.echo("Deleted!")


@APP.command()
def run_pyright() -> None:
    """Generate typestubs GoobAI"""
    typer.echo("Generating type stubs for GoobAI")
    repo_typing.run_pyright()


@APP.command()
def run_screencrop() -> None:
    """Manually run screncrop service and get bounding boxes"""
    try:
        asyncio.run(
            ImageService.bindingbox_handler(
                "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00000.JPEG",
            )
        )
    except Exception as ex:
        print(f"{ex}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"Error Class: {ex.__class__}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        print(output)
        print(f"exc_type: {exc_type}")
        print(f"exc_value: {exc_value}")
        traceback.print_tb(exc_traceback)
        if aiosettings.dev_mode:
            bpdb.pm()


@APP.command()
def run_download_and_predict(
    img_url: str = "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00000.JPEG",
) -> None:
    """Manually run screencrop's download_and_predict service and get bounding boxes"""
    path_to_image_from_cli = fix_path(img_url)
    try:
        ImageService.handle_predict_from_file(path_to_image_from_cli)
    except Exception as ex:
        print(f"{ex}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"Error Class: {ex.__class__}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        print(output)
        print(f"exc_type: {exc_type}")
        print(f"exc_value: {exc_value}")
        traceback.print_tb(exc_traceback)
        if aiosettings.dev_mode:
            bpdb.pm()


@APP.command()
def query_readthedocs() -> None:
    """Smoketest for querying readthedocs pdfs against vectorstore."""
    try:
        import rich

        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        from goob_ai.services.chroma_service import CHROMA_PATH, DATA_PATH, ChromaService
        from goob_ai.utils import file_functions

        client = ChromaService.client
        test_collection_name = "readthedocs"

        documents = []

        d = file_functions.tree(DATA_PATH)
        result = file_functions.filter_pdfs(d)

        for filename in result:
            LOGGER.info(f"Loading document: {filename}")
            db = ChromaService.add_to_chroma(
                path_to_document=f"{filename}",
                collection_name=test_collection_name,
                embedding_function=None,
            )

        embedding_function = OpenAIEmbeddings()

        db = Chroma(
            client=client,
            collection_name=test_collection_name,
            embedding_function=embedding_function,
        )

        # query it
        query = "How do I enable syntax highlighting with rich?"
        docs = db.similarity_search(query)
        rich.print("Answer: ")
        # rich.print(docs)
        print(docs[0].page_content)

    except Exception as ex:
        print(f"{ex}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"Error Class: {ex.__class__}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        print(output)
        print(f"exc_type: {exc_type}")
        print(f"exc_value: {exc_value}")
        traceback.print_tb(exc_traceback)
        if aiosettings.dev_mode:
            bpdb.pm()


@APP.command()
def run_predict_and_display(img_url: list[str] = None) -> None:
    """Manually run screencrop's download_and_predict service and get bounding boxes"""
    if img_url is None:
        img_url = [  # type: ignore
            # "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger.JPEG",
            # "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger2.PNG",
            # "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger3.PNG",
            "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00000.JPEG",
            "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00001.PNG",
            "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00002.PNG",
            "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00003.PNG",
            "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00004.PNG",
            "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00005.PNG",
            "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00006.PNG",
            "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00007.PNG",
            "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00008.PNG",
            "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00009.PNG",
            # "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00010.PNG",
            # "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00011.PNG",
            # "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00012.PNG",
            # "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00013.PNG",
        ]
    path_to_image_from_cli = fix_path(img_url)
    try:
        ImageService.handle_predict_and_display(path_to_image_from_cli)
    except Exception as ex:
        print(f"{ex}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"Error Class: {ex.__class__}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        print(output)
        print(f"exc_type: {exc_type}")
        print(f"exc_value: {exc_value}")
        traceback.print_tb(exc_traceback)
        if aiosettings.dev_mode:
            bpdb.pm()


# THIS SHOULD BE THE FINAL ONE THAT PRODUCES THE PROPER CROP
@APP.command()
def run_final() -> None:
    """Manually run screencrop's download_and_predict service and get bounding boxes"""
    img_path = "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00013.PNG"
    path_to_image_from_cli = fix_path(img_path)
    try:
        ImageService.handle_final(path_to_image_from_cli)
    except Exception as ex:
        print(f"{ex}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"Error Class: {ex.__class__}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        print(output)
        print(f"exc_type: {exc_type}")
        print(f"exc_value: {exc_value}")
        traceback.print_tb(exc_traceback)
        if aiosettings.dev_mode:
            bpdb.pm()


# THIS SHOULD BE THE FINAL ONE THAT PRODUCES THE PROPER CROP
@APP.command()
def chroma(choices: ChromaChoices) -> None:
    """Interact w/ chroma local vectorstore"""
    try:
        if choices == ChromaChoices.load:
            ChromaService.load_documents()
        elif choices == ChromaChoices.generate:
            ChromaService.generate_data_store()
        elif choices == ChromaChoices.get_response:
            ChromaService.get_response("hello")
    except Exception as ex:
        print(f"{ex}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"Error Class: {ex.__class__}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        print(output)
        print(f"exc_type: {exc_type}")
        print(f"exc_value: {exc_value}")
        traceback.print_tb(exc_traceback)
        if aiosettings.dev_mode:
            bpdb.pm()


@APP.command()
def go() -> None:
    """Main entry point for GoobAI"""
    typer.echo("Starting up GoobAI Bot")
    asyncio.run(run_bot())


if __name__ == "__main__":
    APP()
