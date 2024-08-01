"""goob_ai.gen_ai.tools.download_tool"""

# mypy: disable-error-code="arg-type"
# mypy: disable-error-code="attr-defined"
# mypy: disable-error-code="call-arg"
# mypy: disable-error-code="index"
# mypy: disable-error-code="misc"
# mypy: disable-error-code="return"
# mypy: disable-error-code="union-attr"
# pylint: disable=no-member
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import functools
import logging
import os
import os.path
import pathlib
import re
import sys
import tempfile
import time
import traceback
import typing
import uuid

from enum import IntEnum
from timeit import default_timer as timer
from typing import Any, ClassVar, Dict, List, NewType, Optional, Type

import aiohttp
import discord
import openai
import requests
import rich
import uritools

from codetiming import Timer
from discord.ext import commands
from discord.message import Message
from discord.user import User
from langchain.pydantic_v1 import BaseModel, Field
from langchain.pydantic_v1 import BaseModel as BaseModelV1
from langchain.pydantic_v1 import Field as FieldV1
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.runnables import ConfigurableField, Runnable, RunnableBranch, RunnableLambda, RunnableMap
from langchain_core.tools import BaseTool, ToolException
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from loguru import logger as LOGGER
from openai import Client
from pydantic import BaseModel

from goob_ai import db, helpers, shell, utils
from goob_ai.aio_settings import aiosettings
from goob_ai.clients.http_client import HttpClient
from goob_ai.constants import (
    FIFTY_THOUSAND,
    FIVE_HUNDRED_THOUSAND,
    MAX_BYTES_UPLOAD_DISCORD,
    ONE_HUNDRED_THOUSAND,
    ONE_MILLION,
    TEN_THOUSAND,
    THIRTY_THOUSAND,
    TWENTY_THOUSAND,
)
from goob_ai.utils import vidops
from goob_ai.utils.file_functions import VIDEO_EXTENSIONS, get_files_to_upload, unlink_orig_file


# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
class DownloadToolInput(BaseModelV1):
    """
    Use this tool to download images and video from twitter using gallery-dl. Use whenever the url comes from domain x.com or twitter.com or some subset of those DNS names.

    Args:
        url: url to read
    """

    url: str = FieldV1(..., title="url", description="url to read")
    # url: str = Field(..., description="The URL to the image file.")
    # prompt: str = Field(..., description="The prompt to use for the API call.")


# class DownloadTool(BaseTool):
#     name: str = "download_api"
#     description: str = "Use this tool to download images and video from twitter using gallery-dl. Use whenever the url comes from domain x.com or twitter.com or some subset of those DNS names."
#     args_schema: type[BaseModel] = DownloadToolInput
#     return_direct: bool = False
#     handle_tool_error: bool = True
#     verbose: bool = True

#     def _run(self, url: str, **kwargs) -> str | bytes:  # type: ignore
#         """
#         Use this tool to download images and video from twitter using glallery-dl. Use whenever the url comes from domain x.com or twitter.com or some subset of those DNS names.

#         Args:
#             url: url to read
#         """

#         LOGGER.info(f"url = {url}")

#         try:
#             dl_uri = uritools.urisplit(url)

#             with tempfile.TemporaryDirectory() as tmpdirname:
#                 print("created temporary directory", tmpdirname)
#                 with Timer(text="\nTotal elapsed time: {:.1f}"):
#                     gallery_dl_cmd = [
#                         "gallery-dl",
#                         "--no-mtime",
#                         "-v",
#                         "--write-info-json",
#                         "--write-metadata",
#                         f"{dl_uri.geturi()}",
#                     ]

#                     ret = shell.pquery(gallery_dl_cmd, cwd=f"{tmpdirname}")

#                     LOGGER.debug(f"Success, downloaded {dl_uri.geturi()}")
#                     # refactor this into a function
#                     file_to_upload = get_files_to_upload(tmpdirname)
#                     LOGGER.debug(f"BEFORE: {type(self).__name__} -> file_to_upload = {file_to_upload}")

#                     needs_compression = False
#                     # if it is a video, we should compress it
#                     for meme in file_to_upload:
#                         res = vidops.compress_video(f"{tmpdirname}", f"{meme}", self.bot, ctx)

#                         # if it compressed at least one video, return true
#                         if res:
#                             # since something was compressed, we want to rerun get_files_to_upload
#                             needs_compression = True
#                         LOGGER.debug(f"AFTER COMPRESSION: {file_to_upload} -> file_to_upload = {file_to_upload}")

#                     # If something was compressed, regenerate file_to_upload var.
#                     if needs_compression:
#                         # NOTE: It's possible this list changed due to compression, verify it.
#                         # refactor this into a function
#                         file_to_upload = get_files_to_upload(tmpdirname)

#                     LOGGER.debug(f"AFTER: {type(self).__name__} -> file_to_upload = {file_to_upload}")

#                     for meme in file_to_upload:
#                         # if it is a video, compress it
#                         if pathlib.Path(meme).stat().st_size > MAX_BYTES_UPLOAD_DISCORD:
#                             # await ctx.send(
#                             #     embed=discord.Embed(
#                             #         description=f"File is over 8MB... Uploading to dropbox -> {meme}...."
#                             #     )
#                             # )

#                             to_upload_list = [meme]
#                             # await aiodbx.dropbox_upload(to_upload_list)

#                             # await ctx.send(
#                             #     embed=discord.Embed(description=f"File successfully uploaded to dropbox! -> {meme}....")
#                             # )
#                         else:
#                             # await ctx.send(
#                             #     embed=discord.Embed(description=f"Uploading to discord -> {file_to_upload}....")
#                             # )

#                             my_files = []

#                             for f in file_to_upload:
#                                 rich.print(f)
#                                 my_files.append(discord.File(f"{f}"))

#                             LOGGER.debug(f"{type(self).__name__} -> my_files = {my_files}")
#                             rich.print(my_files)

#                             try:
#                                 msg: Message
#                                 # msg = await ctx.send(files=my_files)
#                                 # await ctx.send(
#                                 #     embed=discord.Embed(description=f"File successfully uploaded -> {my_files}....")
#                                 # )
#                             except Exception as ex:
#                                 # await ctx.send(embed=discord.Embed(description="Could not upload story to discord...."))
#                                 print(ex)
#                                 exc_type, exc_value, exc_traceback = sys.exc_info()
#                                 err_msg = f"Error invoking regular DownloadTool(url='{url}'): exc_type={type(ex).__name__},exc_value='{exc_value}': {ex}"
#                                 LOGGER.error(f"Error Class: {str(ex.__class__)}")
#                                 output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
#                                 LOGGER.warning(output)
#                                 # await ctx.send(embed=discord.Embed(description=output))
#                                 LOGGER.error(f"exc_type: {exc_type}")
#                                 LOGGER.error(f"exc_value: {exc_value}")
#                                 traceback.print_tb(exc_traceback)
#                                 raise ToolException(err_msg) from ex

#         except Exception as e:
#             exc_type, exc_value, exc_traceback = sys.exc_info()
#             err_msg = f"Error invoking regular DownloadTool(url='{url}'): exc_type={type(e).__name__},exc_value='{exc_value}': {e}"
#             LOGGER.error(err_msg)
#             # LOGGER.error(f"exc_type={exc_type},exc_value={exc_value}")
#             # LOGGER.error(f"Args: image_path={image_path}, prompt={prompt}")
#             traceback.print_tb(exc_traceback)
#             raise ToolException(err_msg) from e

#     async def _arun(
#         self,
#         url: str,
#         run_manager: Optional[AsyncCallbackManagerForToolRun] | None = None, # type: ignore
#         **kwargs,
#     ) -> str: # type: ignore
#         """
#         Use this asynchronous tool to download images and video from twitter using glallery-dl. Use whenever the url comes from domain x.com or twitter.com or some subset of those DNS names.

#         Args:
#             url: url to read
#         """

#         LOGGER.info(f"url = {url}")

#         try:
#             dl_uri = uritools.urisplit(url)

#             with tempfile.TemporaryDirectory() as tmpdirname:
#                 print("created temporary directory", tmpdirname)
#                 with Timer(text="\nTotal elapsed time: {:.1f}"):
#                     gallery_dl_cmd = [
#                         "gallery-dl",
#                         "--no-mtime",
#                         "-v",
#                         "--write-info-json",
#                         "--write-metadata",
#                         f"{dl_uri.geturi()}",
#                     ]

#                     ret = await shell._aio_run_process_and_communicate(gallery_dl_cmd, cwd=f"{tmpdirname}")

#                     LOGGER.debug(f"Success, downloaded {dl_uri.geturi()}")
#                     # refactor this into a function
#                     file_to_upload = get_files_to_upload(tmpdirname)
#                     LOGGER.debug(f"BEFORE: {type(self).__name__} -> file_to_upload = {file_to_upload}")

#                     needs_compression = False
#                     # if it is a video, we should compress it
#                     for meme in file_to_upload:
#                         res = await vidops.aio_compress_video(f"{tmpdirname}", f"{meme}", self.bot, ctx)

#                         # if it compressed at least one video, return true
#                         if res:
#                             # since something was compressed, we want to rerun get_files_to_upload
#                             needs_compression = True
#                         LOGGER.debug(f"AFTER COMPRESSION: {file_to_upload} -> file_to_upload = {file_to_upload}")

#                     # If something was compressed, regenerate file_to_upload var.
#                     if needs_compression:
#                         # NOTE: It's possible this list changed due to compression, verify it.
#                         # refactor this into a function
#                         file_to_upload = get_files_to_upload(tmpdirname)

#                     LOGGER.debug(f"AFTER: {type(self).__name__} -> file_to_upload = {file_to_upload}")

#                     for meme in file_to_upload:
#                         # if it is a video, compress it
#                         if pathlib.Path(meme).stat().st_size > MAX_BYTES_UPLOAD_DISCORD:
#                             # await ctx.send(
#                             #     embed=discord.Embed(description=f"File is over 8MB... Uploading to dropbox -> {meme}....")
#                             # )

#                             to_upload_list = [meme]
#                             # await aiodbx.dropbox_upload(to_upload_list)

#                             # await ctx.send(
#                             #     embed=discord.Embed(description=f"File successfully uploaded to dropbox! -> {meme}....")
#                             # )
#                         else:
#                             # await ctx.send(embed=discord.Embed(description=f"Uploading to discord -> {file_to_upload}...."))

#                             my_files = []

#                             for f in file_to_upload:
#                                 rich.print(f)
#                                 my_files.append(discord.File(f"{f}"))

#                             LOGGER.debug(f"{type(self).__name__} -> my_files = {my_files}")
#                             rich.print(my_files)

#                             try:
#                                 msg: Message
#                                 # msg = await ctx.send(files=my_files)
#                                 # await ctx.send(
#                                 #     embed=discord.Embed(description=f"File successfully uploaded -> {my_files}....")
#                                 # )
#                             except Exception as ex:
#                                 # await ctx.send(embed=discord.Embed(description="Could not upload story to discord...."))
#                                 print(ex)
#                                 exc_type, exc_value, exc_traceback = sys.exc_info()
#                                 err_msg = f"Error invoking regular DownloadTool(url='{url}'): exc_type={type(ex).__name__},exc_value='{exc_value}': {ex}"
#                                 LOGGER.error(f"Error Class: {str(ex.__class__)}")
#                                 output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
#                                 LOGGER.warning(output)
#                                 # await ctx.send(embed=discord.Embed(description=output))
#                                 LOGGER.error(f"exc_type: {exc_type}")
#                                 LOGGER.error(f"exc_value: {exc_value}")
#                                 traceback.print_tb(exc_traceback)
#                                 raise ToolException(err_msg) from ex

#         except Exception as e:
#             exc_type, exc_value, exc_traceback = sys.exc_info()
#             err_msg = f"Error invoking regular DownloadTool(url='{url}'): exc_type={type(e).__name__},exc_value='{exc_value}': {e}"
#             LOGGER.error(err_msg)
#             LOGGER.error(f"exc_type={exc_type},exc_value={exc_value}")
#             LOGGER.error(f"Args: url='{url}'")
#             traceback.print_tb(exc_traceback)
#             raise ToolException(err_msg) from e
