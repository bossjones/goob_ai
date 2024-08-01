"""test_bot"""

# pylint: disable=assigning-non-slot
# pylint: disable=consider-using-from-import
from __future__ import annotations

import asyncio
import os

from io import BytesIO
from typing import TYPE_CHECKING

import aiohttp
import discord.ext.test as dpytest
import pytest_asyncio

from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientError
from discord.ext.commands import Cog, command
from goob_ai import aio_settings
from goob_ai.goob_bot import AsyncGoobBot, download_image

import pytest


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture

IS_RUNNING_ON_GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTOR"))


class Misc(Cog):
    @command()
    async def ping(self, ctx):
        await ctx.send("Pong !")

    @command()
    async def echo(self, ctx, text: str):
        await ctx.send(text)


@pytest_asyncio.fixture
# async def bot(mocker: MockerFixture, monkeypatch: MonkeyPatch):
async def bot():
    # monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_TOKEN", "fake_discord_token")
    # monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_TOKEN", "fake_discord_token")
    # monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_ADMIN_USER_ID", 1337)
    # monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_SERVER_ID", 1337)
    # monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_CLIENT_ID", 8008)
    # monkeypatch.setenv("GOOB_AI_CONFIG_OPENAI_API_KEY", "fake_openai_key")
    # monkeypatch.setenv("OPENAI_API_KEY", "fake_openai_key")
    # monkeypatch.setenv("PINECONE_API_KEY", "fake_pinecone_key")
    # monkeypatch.setenv("PINECONE_INDEX", "fake_test_index")

    # test_settings: aio_settings.AioSettings = aio_settings.AioSettings()

    test_bot: AsyncGoobBot = AsyncGoobBot()
    # # Setup
    # intents = discord.Intents.default()
    # intents.members = True
    # intents.message_content = True
    # # b = commands.Bot(command_prefix="!",
    # #                  intents=intents)
    await test_bot._async_setup_hook()  # setup the loop
    await test_bot.add_cog(Misc())

    dpytest.configure(test_bot)

    yield test_bot

    # Teardown
    await dpytest.empty_queue()  # empty the global message queue as test teardown
    # await bot.pool.disconnect()


# @pytest.mark.skipif(
#     os.getenv("GOOB_AI_BOT_SANITY"),  # noqa
#     reason="These tests are meant to only run locally on laptop prior to porting it over to new system",
# )
@pytest.mark.filterwarnings("ignore:unclosed <ssl.SSLSocket ")
@pytest.mark.integration()
class TestBot:
    @pytest.mark.asyncio()
    # @pytest.mark.justaio
    async def test_defaults(self, mocker: MockerFixture, monkeypatch: MonkeyPatch) -> None:
        # pytest-asyncio provides and manages the `event_loop`
        # and it should be set as the default loop for this session
        # assert event_loop  # but it's not == asyncio.get_event_loop() ?
        # paranoid about weird libraries trying to read env vars during testing
        monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_TOKEN", "fake_discord_token")
        monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_TOKEN", "fake_discord_token")
        monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_ADMIN_USER_ID", 1337)
        monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_SERVER_ID", 1337)
        monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_CLIENT_ID", 8008)
        monkeypatch.setenv("GOOB_AI_CONFIG_OPENAI_API_KEY", "fake_openai_key")
        monkeypatch.setenv("OPENAI_API_KEY", "fake_openai_key")
        monkeypatch.setenv("PINECONE_API_KEY", "fake_pinecone_key")
        monkeypatch.setenv("PINECONE_INDEX", "fake_test_index")
        await asyncio.sleep(0.05)

        test_settings: aio_settings.AioSettings = aio_settings.AioSettings()

        test_bot: AsyncGoobBot = AsyncGoobBot()

        assert test_bot.intents.members
        # NOTE: https://github.com/makupi/cookiecutter-discord.py-postgres/blob/master/%7B%7Bcookiecutter.bot_slug%7D%7D/bot/__init__.py
        test_bot.version = "fake"
        test_bot.guild_data = {}

        # # import bpdb

        # # bpdb.set_trace()

        # assert test_bot.version == "fake"
        # assert test_bot.description == "Better than the last one"
        # assert not test_bot.tasks
        # assert test_bot.num_workers == 3
        # assert not test_bot.job_queue

        # # This group of variables are used in the upscaling process
        # assert not test_bot.last_model
        # assert not test_bot.last_in_nc
        # assert not test_bot.last_out_nc
        # assert not test_bot.last_nf
        # assert not test_bot.last_nb
        # assert not test_bot.last_scale
        # assert not test_bot.last_kind
        # assert not test_bot.model
        # assert not test_bot.autocrop_model

        # # NOTE: these are temporary
        # assert (
        #     test_bot.ml_models_path == "/Users/malcolm/dev/universityofprofessorex/cerebro-bot/goob_ai/ml_models/"
        # )

        # assert test_bot.my_custom_ml_models_path == "/Users/malcolm/Downloads/"

        # # import bpdb

        # # bpdb.set_trace()

        # if not torch.cuda.is_available():
        #     assert str(test_bot.device) == "cpu"

        # # await test_bot.metrics_api.stop()


@pytest.mark.integration()
class TestBotWithDPyTest:
    @pytest.mark.asyncio()
    async def test_ping(self, bot):
        await dpytest.message("?ping")
        assert dpytest.verify().message().content("Pong !")

    @pytest.mark.asyncio()
    async def test_echo(self, bot):
        await dpytest.message("?echo Hello world")
        assert dpytest.verify().message().contains().content("Hello")


# @pytest.mark.asyncio
# @pytest.mark.parametrize(
#     "url, status, content, expected",
#     [
#         ("http://example.com/image.jpg", 200, b"image data", b"image data"),
#         ("http://example.com/image.png", 200, b"another image data", b"another image data"),
#         ("http://example.com/image.gif", 200, b"gif image data", b"gif image data"),
#     ],
#     ids=["jpg image", "png image", "gif image"],
# )
# async def test_download_image_happy_path(url, status, content, expected, mocker):
#     # Arrange
#     async def mock_get(*args, **kwargs):
#         mock_response = mocker.AsyncMock()
#         mock_response.status = status
#         mock_response.read.return_value = content
#         return mock_response

#     mocker.patch.object(ClientSession, "get", new=mock_get)
#     # Act
#     result = await download_image(url)

#     # Assert
#     assert isinstance(result, BytesIO)
#     assert result.read() == expected


# @pytest.mark.asyncio
# @pytest.mark.parametrize(
#     "url, status, content, expected_exception",
#     [
#         ("http://example.com/notfound.jpg", 404, b"", None),
#         ("http://example.com/servererror.jpg", 500, b"", None),
#         ("http://example.com/timeout.jpg", 408, b"", None),
#     ],
#     ids=["404 not found", "500 server error", "408 request timeout"],
# )
# async def test_download_image_error_cases(url, status, content, expected_exception, mocker):
#     # Arrange
#     async def mock_get(*args, **kwargs):
#         mock_response = mocker.AsyncMock()
#         mock_response.status = status
#         mock_response.read.return_value = content
#         return mock_response

#     mocker.patch.object(ClientSession, "get", new=mock_get)
#     # Act
#     result = await download_image(url)

#     # Assert
#     assert result is None


# @pytest.mark.asyncio
# @pytest.mark.parametrize(
#     "url, exception, expected_exception",
#     [
#         ("http://example.com/connectionerror.jpg", ClientError, ClientError),
#         ("http://example.com/invalidurl.jpg", ValueError, ValueError),
#     ],
#     ids=["client error", "invalid URL"],
# )
# async def test_download_image_exceptions(url, exception, expected_exception, mocker):
#     # Arrange
#     async def mock_get(*args, **kwargs):
#         raise exception

#     mocker.patch.object(ClientSession, "get", new=mock_get)
#     # Act & Assert
#     with pytest.raises(expected_exception):
#         await download_image(url)
